import cv2
import numpy as np
from src.image_processing import load_image
from src.segmentation import PieceDetector
from src.matching import ReferenceAnalyzer, PieceMatcher

def draw_matches(ref_image, matches):
    """
    Draws the matched pieces' bounding boxes on the reference image.
    matches: List of tuples (piece_index, homography)
    """
    output = ref_image.copy()
    
    for piece_idx, H, piece_info in matches:
        # Get piece dimensions (from bbox w, h)
        # We need the original piece contour or bbox to project it.
        # Let's project a simple box representing the piece frame.
        h, w = piece_info['bbox'][3], piece_info['bbox'][2]
        
        # Define corners of the piece in its local coordinate system
        # (Assuming the piece image passed to matcher was the crop)
        # The keypoints were detected in the crop.
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        
        try:
            dst = cv2.perspectiveTransform(pts, H)
            cv2.polylines(output, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
            
            # Draw centroid or label
            M = cv2.moments(dst)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(output, str(piece_idx), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        except Exception as e:
            print(f"Error drawing match for piece {piece_idx}: {e}")
            
    return output

def overlay_piece(bg_img, piece_alpha, H):
    h_bg, w_bg = bg_img.shape[:2]
    h_p, w_p = piece_alpha.shape[:2]
    
    # Warp the piece image (4 channels)
    warped_piece = cv2.warpPerspective(piece_alpha, H, (w_bg, h_bg))
    
    # Extract alpha mask
    alpha = warped_piece[:, :, 3] / 255.0
    
    # Overlay
    for c in range(3):
        bg_img[:, :, c] = (alpha * warped_piece[:, :, c] + (1 - alpha) * bg_img[:, :, c]).astype(np.uint8)
        
    return bg_img

def polygons_overlap(H1, H2, shape1, shape2):
    # Check if two placed pieces overlap significantly
    h1, w1 = shape1[:2]
    h2, w2 = shape2[:2]
    
    pts1 = np.float32([[0,0], [0,h1], [w1,h1], [w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0], [0,h2], [w2,h2], [w2,0]]).reshape(-1,1,2)
    
    dst1 = cv2.perspectiveTransform(pts1, H1)
    dst2 = cv2.perspectiveTransform(pts2, H2)
    
    # Create masks and calculate intersection
    # Just checking bounding box intersection first for speed
    rect1 = cv2.boundingRect(np.int32(dst1))
    rect2 = cv2.boundingRect(np.int32(dst2))
    
    # rect is x, y, w, h
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    
    if x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1:
        return False # No bbox overlap
        
    # Calculate more exact overlap area if bbox overlaps
    # Create simple binary masks
    mask1 = np.zeros((1000, 1000), dtype=np.uint8) # Arbitrary large enough size, or max of rects
    # Actually, proper way is polygon intersection area, but let's just use shapely or opencv
    # For simplicity, we can just say if centers are too close or bounding rects overlap significantly
    
    overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    overlap_area = overlap_x * overlap_y
    
    area1 = w1 * h1
    area2 = w2 * h2
    
    # If overlap is more than 30% of either piece
    if overlap_area > 0.3 * min(area1, area2):
        return True
    return False

def main():
    # 1. Load Images
    print("Loading images...")
    try:
        box_image = load_image('data/box.jpg')
        pieces_image = load_image('data/pieces.jpg')
    except Exception:
        box_image = load_image('puzzle_solver/data/box.jpg')
        pieces_image = load_image('puzzle_solver/data/pieces.jpg')
    
    # 2. Segment Pieces
    print("Segmenting pieces...")
    detector = PieceDetector()
    pieces = detector.detect_pieces(pieces_image)
    counts = {'interior': 0, 'edge': 0, 'corner': 0}
    for p in pieces:
        counts[p.get('type', 'interior')] += 1
    print(f"Found {len(pieces)} pieces: {counts['corner']} corners, {counts['edge']} edges, {counts['interior']} interior.")
    
    # 3. Analyze Reference
    print("Analyzing reference...")
    ref_analyzer = ReferenceAnalyzer(box_image)
    kp, des = ref_analyzer.compute_features()
    print(f"Reference features: {len(kp)}")
    
    # 4. Match Pieces (Independent)
    print("Matching pieces independently...")
    matcher = PieceMatcher(ref_analyzer)
    
    all_matches = []
    
    for idx, piece in enumerate(pieces):
        # Reset features for each piece so they all match against the full reference
        matcher.reset_features()
        success, H, info = matcher.match_piece(piece, min_matches=6)
        
        if success:
            inliers = info.get('inliers', 0)
            
            # 2b. Geometric Verification (Scale and Distortion)
            h, w = piece['image'].shape[:2]
            pts = np.float32([[0,0], [0,h], [w,h], [w,0]]).reshape(-1,1,2)
            dst_pts = cv2.perspectiveTransform(pts, H)
            
            # Validate geometric distortion rigorously
            invalid_geometry = False
            for pt in dst_pts:
                # Sanity bounds check
                if pt[0][0] < -5000 or pt[0][0] > 10000 or pt[0][1] < -5000 or pt[0][1] > 10000:
                    invalid_geometry = True
                    break
            
            if invalid_geometry: continue
            
            piece_area = w * h
            dst_area = cv2.contourArea(dst_pts)
            if piece_area == 0: continue
            scale_ratio = dst_area / piece_area
            
            # Tighter scale constraint - puzzles shouldn't scale drastically
            if scale_ratio < 1.0 or scale_ratio > 10.0:
                continue
                
            if not cv2.isContourConvex(np.int32(dst_pts)):
                continue
                
            # Check aspect ratio distortion
            rect = cv2.minAreaRect(dst_pts)
            rw, rh = rect[1]
            if rw == 0 or rh == 0: continue
            ratio = max(rw/rh, rh/rw)
            if ratio > 3.0: continue # Pieces shouldn't become long rectangles
            
            if inliers >= 5:
                all_matches.append({
                    'piece_idx': idx,
                    'piece': piece,
                    'H': H,
                    'inliers': inliers,
                    'scale': scale_ratio
                })
                print(f"Piece {idx} ({piece.get('type', 'interior')}) broadly matched! Inliers: {inliers}, Scale: {scale_ratio:.2f}")

    # 5. Non-Maximum Suppression (Sort by inliers)
    print("\nResolving overlapping matches...")
    all_matches.sort(key=lambda x: x['inliers'], reverse=True)
    
    final_matches = []
    
    for match in all_matches:
        idx = match['piece_idx']
        H = match['H']
        piece = match['piece']
        
        # Check against already placed pieces
        overlaps = False
        for placed in final_matches:
            if polygons_overlap(H, placed['H'], piece['image'].shape, placed['piece']['image'].shape):
                overlaps = True
                break
                
        if not overlaps:
            final_matches.append(match)
            print(f"Placed Piece {idx} with {match['inliers']} inliers.")
        else:
            print(f"Discarded Piece {idx} due to overlap.")
            
    print(f"\nMatch complete. Found {len(final_matches)} solid unique placements.")
    
    # 6. Build the Recursive Assembled Canvas!
    print("\n--- Assembling Puzzle ---")
    h_bg, w_bg = ref_analyzer.reference_image.shape[:2]
    canvas_h, canvas_w = h_bg * 3, w_bg * 3
    
    # We create a dark solid canvas dynamically sized
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 30
    
    # Because pieces can map structurally off the edges of the limited reference box cover,
    # we explicitly place 'data/box.jpg' in the direct center of the large canvas natively.
    offset_y, offset_x = h_bg, w_bg
    T = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float64)
    
    # User Request: Final output is also saved without mapping on the box cover
    # canvas[offset_y:offset_y+h_bg, offset_x:offset_x+w_bg] = ref_analyzer.reference_image
    
    placed_matches = []
    
    for match in final_matches:
        idx = match['piece_idx']
        piece = match['piece']
        H = match['H']
        
        # Shift Homography to translated native Canvas
        H_shifted = T @ H
        canvas = overlay_piece(canvas, piece['image_alpha'], H_shifted)
        match['H'] = H_shifted
        placed_matches.append(match)
        
    cv2.imwrite('assembled_step_0.jpg', canvas)
    
    unmatched_indices = [i for i in range(len(pieces)) if i not in [m['piece_idx'] for m in placed_matches]]
    
    iteration = 1
    max_assemble_iters = 5
    
    while unmatched_indices and iteration <= max_assemble_iters:
        print(f"\nAssembly Iteration {iteration}: {len(unmatched_indices)} pieces left...")
        
        # 7. Feed the newly structurally synthesized canvas back into the solver as the Reference Map!!
        canvas_analyzer = ReferenceAnalyzer(canvas)
        canvas_analyzer.compute_features()
        canvas_matcher = PieceMatcher(canvas_analyzer)
        print(f"Total Structural Canvas features: {len(canvas_analyzer.keypoints)}")
        
        newly_placed = []
        
        for idx in unmatched_indices:
            piece = pieces[idx]
            canvas_matcher.reset_features()
            # Match directly onto the blended visual edges!
            success, H_shifted, info = canvas_matcher.match_piece(piece, min_matches=6)
            
            if success:
                inliers = info.get('inliers', 0)
                
                h, w = piece['image'].shape[:2]
                pts = np.float32([[0,0], [0,h], [w,h], [w,0]]).reshape(-1,1,2)
                dst_pts = cv2.perspectiveTransform(pts, H_shifted)
                
                # Validate geometric distortion rigorously
                invalid_geometry = False
                for pt in dst_pts:
                    # Sanity bounds check
                    if pt[0][0] < -5000 or pt[0][0] > 10000 or pt[0][1] < -5000 or pt[0][1] > 10000:
                        invalid_geometry = True
                        break
                
                if invalid_geometry: continue
                
                piece_area = w * h
                dst_area = cv2.contourArea(dst_pts)
                if piece_area == 0: continue
                scale_ratio = dst_area / piece_area
                
                    
                # Strict color geometric evaluation evaluating overlap bounding maps 
                h_ref, w_ref = canvas.shape[:2]
                piece_mask = piece['mask']
                warped_mask = cv2.warpPerspective(piece_mask, H_shifted, (w_ref, h_ref))
                
                # Exclude mapping into empty space directly exclusively
                intersection_canvas = cv2.bitwise_and(canvas[:,:,0], canvas[:,:,0], mask=warped_mask)
                non_empty = cv2.countNonZero(intersection_canvas)
                
                # Ensure the bound connects computationally
                if non_empty < 0.1 * piece_area:
                    continue
                if not cv2.isContourConvex(np.int32(dst_pts)):
                    continue
                    
                # Check aspect ratio distortion
                rect = cv2.minAreaRect(dst_pts)
                rw, rh = rect[1]
                if rw == 0 or rh == 0: continue
                ratio = max(rw/rh, rh/rw)
                if ratio > 3.0: continue # Pieces shouldn't become long rectangles
                    
                if inliers >= 8: # Confidence bounding
                    newly_placed.append({
                        'piece_idx': idx,
                        'piece': piece,
                        'H': H_shifted,
                        'inliers': inliers,
                        'scale': scale_ratio
                    })
                    print(f"Piece {idx} structurally matched to Canvas Edge! Inliers: {inliers}, Scale: {scale_ratio:.2f}")

        # NMS on strictly new matches
        newly_placed.sort(key=lambda x: x['inliers'], reverse=True)
        
        placed_this_round = 0
        for match in newly_placed:
            idx = match['piece_idx']
            H = match['H']
            piece = match['piece']
            
            overlaps = False
            for placed in placed_matches:
                if polygons_overlap(H, placed['H'], piece['image'].shape, placed['piece']['image'].shape):
                    overlaps = True
                    break
                    
            if not overlaps:
                placed_matches.append(match)
                canvas = overlay_piece(canvas, piece['image_alpha'], H)
                unmatched_indices.remove(idx)
                placed_this_round += 1
                print(f"Placed Piece {idx} stably with {match['inliers']} inliers onto Canvas.")
            
        cv2.imwrite(f'assembled_step_{iteration}.jpg', canvas)
        if placed_this_round == 0:
            print("Mathematical deadlock limit reached. No native pieces matched further.")
            break
            
        iteration += 1

    print(f"\nAssembly process recursively complete. Placed {len(placed_matches)} / {len(pieces)} uniquely bound shapes.")
    cv2.imwrite('solution_assembled.jpg', canvas)
    print("Saved fully assembled robust output to 'solution_assembled.jpg'.")

if __name__ == "__main__":
    main()
