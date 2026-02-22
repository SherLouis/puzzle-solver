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

def main():
    # 1. Load Images
    print("Loading images...")
    box_image = load_image('puzzle_solver/data/box.jpg')
    pieces_image = load_image('puzzle_solver/data/pieces.jpg')
    
    # 2. Segment Pieces
    print("Segmenting pieces...")
    detector = PieceDetector() # Using robust grabcut defaults
    pieces = detector.detect_pieces(pieces_image)
    print(f"Found {len(pieces)} pieces.")
    
    # 3. Analyze Reference
    print("Analyzing reference...")
    ref_analyzer = ReferenceAnalyzer(box_image)
    kp, des = ref_analyzer.compute_features()
    print(f"Reference features: {len(kp)}")
    
    # 4. Match Pieces
    print("Matching pieces (Iterative)...")
    matcher = PieceMatcher(ref_analyzer)
    
    unmatched_pieces = [i for i in range(len(pieces))]
    final_matches = []
    
    # Mask to track occupied areas (visual only)
    ref_h, ref_w = ref_analyzer.reference_image.shape[:2]
    occupied_mask = np.zeros((ref_h, ref_w), dtype=np.uint8)
    
    iteration = 0
    max_iterations = len(pieces) + 5 # Safety limit
    
    while unmatched_pieces and iteration < max_iterations:
        print(f"\nIteration {iteration}: Matching {len(unmatched_pieces)} pieces against {len(matcher.current_keypoints)} features...")
        iteration += 1
        
        best_match = None
        best_score = -1
        best_idx = -1
        
        # 1. Match all remaining pieces
        for idx in unmatched_pieces:
            piece = pieces[idx]
            success, H, info = matcher.match_piece(piece)
            
            if success:
                score = info.get('inliers', 0)
                # Prioritize strictly higher score
                if score > best_score:
                    best_score = score
                    best_match = (idx, piece, H, info)
        
        # 2. Check if we found ANY match
        if best_match is None or best_score < 4: # Min inliers threshold
            print("No more valid matches found.")
            break
            
        idx, piece, H, info = best_match
        
        # 2b. Geometric Verification (Scale and Distortion)
        # Decompose homography to check scale
        # Simple check: Map corners and check area ratio
        h, w = piece['image'].shape[:2]
        pts = np.float32([[0,0], [0,h], [w,h], [w,0]]).reshape(-1,1,2)
        dst_pts = cv2.perspectiveTransform(pts, H)
        
        # Check area
        piece_area = w * h
        dst_area = cv2.contourArea(dst_pts)
        
        if piece_area == 0: scale_ratio = 0
        else: scale_ratio = dst_area / piece_area
        
        # Valid scale: Piece shouldn't grow > 4x or shrink < 0.25x
        # Unless reference is much larger. 
        # Ref image is ~1000px wide. Piece is ~50px.
        # Scale should be roughly consistent.
        # Let's assume consistent scale across all matches.
        # But for now, just filter extreme nonsense.
        if scale_ratio < 0.1 or scale_ratio > 10.0:
            print(f"  -> Rejected Piece {idx} (Score: {best_score}): Invalid Scale {scale_ratio:.2f}")
            unmatched_pieces.remove(idx) # Don't try again? Or just temporary skip?
            # If we remove it, we won't try it again. True.
            continue

        # Check convexity / distortion
        if not cv2.isContourConvex(np.int32(dst_pts)):
             print(f"  -> Rejected Piece {idx} (Score: {best_score}): Non-convex shape")
             unmatched_pieces.remove(idx)
             continue

        print(f"  -> Accepted Piece {idx} (Score: {best_score}) Scale: {scale_ratio:.2f}")
        final_matches.append((piece, H))
        unmatched_pieces.remove(idx)
        
        # 3. Remove used features from Reference (Masking)
        # Use Mask based removal for robustness
        removal_mask = np.zeros((ref_h, ref_w), dtype=np.uint8)
        cv2.fillConvexPoly(removal_mask, np.int32(dst_pts), 255)
        
        # Filter keypoints using mask
        valid_indices = []
        removed_count = 0
        
        # Vectorized check if possible?
        # matcher.current_keypoints is a list.
        # We can perform batch lookup if we convert to points.
        # But python loop is okay for 2000 points.
        
        kp_points = cv2.KeyPoint_convert(matcher.current_keypoints)
        if len(kp_points) > 0:
            kp_int = np.int32(kp_points)
            # Clip coordinates to image bounds
            kp_int[:, 0] = np.clip(kp_int[:, 0], 0, ref_w - 1)
            kp_int[:, 1] = np.clip(kp_int[:, 1], 0, ref_h - 1)
            
            # Sample the mask
            mask_values = removal_mask[kp_int[:, 1], kp_int[:, 0]]
            
            # Keep indices where mask is 0
            valid_indices = np.where(mask_values == 0)[0].tolist()
            removed_count = len(matcher.current_keypoints) - len(valid_indices)

        print(f"  -> Removed {removed_count} features covered by Piece {idx}")
        matcher.update_features(valid_indices)
        
        # If too few features left, break
        if len(matcher.current_keypoints) < 5:
            print("Reference features exhausted.")
            break

    print(f"\nMatch complete. Found {len(final_matches)} unique placements.")
    
    # Draw matches
    result_img = ref_analyzer.reference_image.copy()
    for piece, H in final_matches:
        h, w = piece['image'].shape[:2]
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, H)
        cv2.polylines(result_img, [np.int32(dst)], True, (0, 255, 0), 3)
        M_moments = cv2.moments(np.int32(dst))
        if M_moments['m00'] != 0:
            cx = int(M_moments['m10'] / M_moments['m00'])
            cy = int(M_moments['m01'] / M_moments['m00'])
            cv2.putText(result_img, "P", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
    cv2.imwrite('solution_result.jpg', result_img)
    print("Saved solution result to 'solution_result.jpg'.")


if __name__ == "__main__":
    main()
