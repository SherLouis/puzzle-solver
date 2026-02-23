import cv2
import numpy as np
import math
from src.image_processing import load_image
from src.segmentation import PieceDetector

def classify_piece_type(contour, idx):
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    
    # We need at least 3 points for defects
    if len(hull_indices) < 3: return 'interior'
    
    try:
        defects = cv2.convexityDefects(contour, hull_indices)
    except cv2.error:
        return 'interior'
        
    if defects is None:
        return 'interior'
        
    # Get the bounding box to know the approximate side lengths
    rect = cv2.minAreaRect(contour)
    # The dimensions of the piece
    w, h = rect[1]
    avg_side = (w + h) / 2
    
    # Filter defects: we only care about "deep" defects (holes or the indentations beside knobs)
    # A standard jigsaw piece has 4 sides. 
    # A straight edge has NO deep defect along its length.
    
    # Let's approximate the polygon to find the main corners
    epsilon = 0.05 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Number of deep defects
    # Actually, a better way is to see how much of the contour perimeter corresponds to the convex hull precisely.
    hull = cv2.convexHull(contour)
    
    # Draw logic to compute long hull segments that match the contour
    contour_len = len(contour)
    straight_sides = 0
    
    # We will identify the 4 sides.
    # We have defects: start, end, far, dist.
    # dist is in 1/256th of a pixel.
    
    # We check the length of contour between 'end' of previous defect and 'start' of next defect.
    # Wait, the points between 'start' and 'end' of a defect are the indented part.
    # The points OUTSIDE the defects are the convex parts (knobs or straight edges).
    
    defect_regions = []
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        depth = d / 256.0
        if depth > avg_side * 0.10: # Significant defect (more than 10% of piece size)
            defect_regions.append((s, e, f, depth))
            
    # Sort defects by start index
    defect_regions.sort(key=lambda x: x[0])
    
    # If a piece is interior, it has 4 sides. Each side is either a knob or a hole.
    # A knob has 2 deep defects adjacent to it (one on each side of the knob base).
    # A hole has 1 deep defect.
    # A straight edge has 0 deep defects AND is relatively long (approx length of the side).
    
    # A simpler approach: A straight edge doesn't deviate from a straight line.
    box = cv2.boxPoints(rect)
    
    straight_edges = 0
    contour_pts = [pt[0] for pt in contour]
    
    for i in range(4):
        p1 = box[i]
        p2 = box[(i+1)%4]
        edge_length = np.linalg.norm(p1 - p2)
        if edge_length < avg_side * 0.5: continue
            
        edge_dir = (p2 - p1) / edge_length
        normal = np.array([-edge_dir[1], edge_dir[0]])
        
        # Calculate distances of all points to this bounding box edge
        # distance from a point to a line
        dy = p2[1] - p1[1]
        dx = p2[0] - p1[0]
        c = p2[0]*p1[1] - p2[1]*p1[0]
        
        close_pts = 0
        for pt in contour_pts:
            dist = abs(dy*pt[0] - dx*pt[1] + c) / edge_length
            if dist < 5.0:
                close_pts += 1
                
        # For a straight edge, the number of contour points very close to the minAreaRect bounding line 
        # should be comparable to the edge length
        # Because the bounding box hugs the outermost points. If an entire side is straight, the bounding box edge will lie perfectly on it.
        # If it's a knob, only the tip of the knob touches the bounding box (very few points).
        print(f"  side {i}: len={edge_length:.1f}, close={close_pts}, ratio={close_pts/edge_length:.2f}")
        if close_pts > 0.4 * edge_length:
            straight_edges += 1
            
    print(f"Piece {idx}: {straight_edges} straight edges")
    if straight_edges >= 2: return 'corner'
    elif straight_edges == 1: return 'edge'
    return 'interior'

img = load_image('puzzle_solver/data/pieces.jpg')
d = PieceDetector()
pieces = d.detect_pieces(img)

counts = {'interior': 0, 'edge': 0, 'corner': 0}
for idx, p in enumerate(pieces):
    cat = classify_piece_type(p['contour'], idx)
    counts[cat] += 1
print(counts)
