import cv2
import numpy as np
import os
from src.image_processing import load_image
from src.segmentation import PieceDetector

def extract_sides(piece):
    contour = piece['contour']
    img = piece['image_alpha']
    center = piece['center']
    
    # 1. Approximate contour to find 4 corners robustly
    peri = cv2.arcLength(contour, True)
    # We use a polygon approximation to find corners.
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    
    # Fallback if approx doesn't yield 4 points.
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int64(box)
    
    contour_pts = [pt[0] for pt in contour]
    
    corners_idx = []
    # Find the contour point closest to each bounding box corner
    for bp in box:
        min_dist = float('inf')
        best_idx = 0
        for i, pt in enumerate(contour_pts):
            dist = np.linalg.norm(pt - bp)
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        corners_idx.append(best_idx)
        
    corners_idx.sort()
    
    sides = []
    n_pts = len(contour_pts)
    for i in range(4):
        start_idx = corners_idx[i]
        end_idx = corners_idx[(i + 1) % 4]
        
        if start_idx < end_idx:
            side_pts = contour_pts[start_idx:end_idx+1]
        else:
            side_pts = contour_pts[start_idx:] + contour_pts[:end_idx+1]
            
        p1 = contour_pts[start_idx]
        p2 = contour_pts[end_idx]
        length = np.linalg.norm(p1 - p2)
        
        if length == 0: continue
        
        edge_dir = (p2 - p1) / length
        normal = np.array([-edge_dir[1], edge_dir[0]])
        
        dists = []
        for pt in side_pts:
            v = pt - p1
            dist = np.dot(v, normal)
            dists.append(dist)
            
        max_dist = max(dists) if dists else 0
        min_dist = min(dists) if dists else 0
        
        # Calculate deviation amplitude
        deviation = max_dist - min_dist
        
        if deviation < length * 0.15:
            # Straight edge
            side_type = 'flat'
        else:
            # Check orientation to determine knob or hole
            v_center = center - p1
            center_dist = np.dot(v_center, normal)
            
            # If the max deviation is away from the piece center, it's a knob
            # If the max deviation is towards the piece center, it's a hole
            # Since normal might be pointing either way, we verify with center dot normal
            if np.sign(max_dist if abs(max_dist) > abs(min_dist) else min_dist) == np.sign(center_dist):
                side_type = 'hole'
            else:
                side_type = 'knob'
                
        # Sample points to create a simplified shape signature of length 50
        sampled_signature = np.zeros(50)
        sampled_colors = []
        
        if len(dists) > 0:
            indices = np.linspace(0, len(dists) - 1, 50).astype(int)
            sampled_signature = np.array(dists)[indices] / length # Normalize by length
            
            # We also want color along the boundary
            for idx in indices:
                pt = side_pts[idx]
                lx, ly = int(pt[0] - piece['bbox'][0]), int(pt[1] - piece['bbox'][1])
                # Ensure within bounds
                lx = max(0, min(lx, img.shape[1] - 1))
                ly = max(0, min(ly, img.shape[0] - 1))
                color = img[ly, lx, :3] # BGR
                sampled_colors.append(color)
        
        sides.append({
            'type': side_type,
            'length': length,
            'signature': sampled_signature,
            'colors': np.array(sampled_colors)
        })
        
    return sides

def match_sides(side_a, side_b):
    # Rule 1: Must be inverse pairs
    if side_a['type'] == 'flat' or side_b['type'] == 'flat':
        return 0, 0
    if side_a['type'] == side_b['type']: # knob-knob or hole-hole
        return 0, 0
        
    # Rule 2: lengths must be roughly equal (within 20%)
    len_ratio = side_a['length'] / side_b['length']
    if len_ratio < 0.8 or len_ratio > 1.2:
        return 0, 0
        
    # Rule 3: Shape Signature match (MSE)
    # The signature of B should be the reverse of A if they are matching.
    # Because A traces clockwise and B traces counter-clockwise when they touch
    sig_b_rev = -side_b['signature'][::-1] # inverse sign and reversed
    shape_diff = np.mean((side_a['signature'] - sig_b_rev) ** 2)
    
    # Rule 4: Color Boundary match (MSE)
    col_b_rev = side_b['colors'][::-1]
    color_diff = np.mean((side_a['colors'] - col_b_rev) ** 2)
    
    # Return match score (lower difference is better)
    return shape_diff, color_diff

def main():
    print("Loading images...")
    img = load_image('puzzle_solver/data/pieces.jpg')
    detector = PieceDetector()
    pieces = detector.detect_pieces(img)
    
    print(f"Total pieces segmented: {len(pieces)}")
    
    # Extract sides for all pieces
    border_pieces = []
    for idx, p in enumerate(pieces):
        sides = extract_sides(p)
        p['sides'] = sides
        
        flat_count = sum(1 for s in sides if s['type'] == 'flat')
        if flat_count > 0:
            border_pieces.append((idx, p))
            
    print(f"Identified {len(border_pieces)} border pieces.")
    
    print("\n--- Testing Pairwise Contour Matches on Border Pieces ---")
    best_matches = []
    
    # O(N^2) pairwise check across border pieces
    for i in range(len(border_pieces)):
        for j in range(i + 1, len(border_pieces)):
            idx_a, piece_a = border_pieces[i]
            idx_b, piece_b = border_pieces[j]
            
            for side_idx_a, side_a in enumerate(piece_a['sides']):
                for side_idx_b, side_b in enumerate(piece_b['sides']):
                    shape_diff, color_diff = match_sides(side_a, side_b)
                    
                    # Heuristics: shape MSE < 0.05, Color MSE < 5000 (roughly 70 per channel)
                    if shape_diff > 0 and shape_diff < 0.05 and color_diff < 5000:
                        best_matches.append({
                            'score': shape_diff * 1000 + color_diff, # combined metric
                            'shape': shape_diff,
                            'color': color_diff,
                            'idx_a': idx_a,
                            'side_a': side_idx_a,
                            'type_a': side_a['type'],
                            'idx_b': idx_b,
                            'side_b': side_idx_b,
                            'type_b': side_b['type']
                        })
                        
    # Sort closest matches
    best_matches.sort(key=lambda x: x['score'])
    
    for i, match in enumerate(best_matches[:10]):
        print(f"Match {i+1}: Piece {match['idx_a']} ({match['type_a']}) and Piece {match['idx_b']} ({match['type_b']})")
        print(f"      Shape Diff: {match['shape']:.4f}, Color MSE: {match['color']:.1f}")

if __name__ == "__main__":
    main()
