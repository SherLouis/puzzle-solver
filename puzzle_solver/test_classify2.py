import cv2
import numpy as np
import math
from src.image_processing import load_image
from src.segmentation import PieceDetector

def classify_piece_type(contour):
    # Find the 4 corners using a rotated rectangle
    rect = cv2.minAreaRect(contour)
    # The dimensions of the piece
    width, height = rect[1]
    
    # Calculate area and perimeter
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # A circularity or compactness metric 4*pi*area / perimeter^2
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    # To find straight edges, we can compute the convexity defects
    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)
    
    # We can categorize sides by walking the contour.
    # But a simpler way to find if a side is straight is to
    # find the best fit rectangle and see if the contour stays very close to the sides of the max inscribed polygon, or just use the bounding box.
    # Let's project contour points onto the 4 edges of the rotated bounding box.
    box = cv2.boxPoints(rect)
    box = np.int64(box)
    
    straight_edges = 0
    contour_pts = [pt[0] for pt in contour]
    
    for i in range(4):
        p1 = box[i]
        p2 = box[(i+1)%4]
        edge_length = np.linalg.norm(p1 - p2)
        if edge_length < 20: continue
            
        # Vector of the edge
        edge_dir = (p2 - p1) / edge_length
        # Normal to the edge
        normal = np.array([-edge_dir[1], edge_dir[0]])
        
        # We find points that project onto this edge segment and measure their distance to the edge.
        # Actually, if an edge is a straight puzzle border, the variation in the distance to the edge will be small, mostly 0, NO knobs and NO holes.
        max_dist_outward = 0
        max_dist_inward = 0
        
        points_on_this_side = False
        
        for pt in contour_pts:
            # Vector from p1 to pt
            v = pt - p1
            # Projection along the edge
            proj = np.dot(v, edge_dir)
            # Distance from the edge (projection along normal)
            dist = np.dot(v, normal)
            
            # If the point projects within the middle 60% of the edge length
            if 0.2 * edge_length < proj < 0.8 * edge_length:
                points_on_this_side = True
                if dist > max_dist_outward:
                    max_dist_outward = dist
                if dist < max_dist_inward:
                    max_dist_inward = dist
                    
        # A straight edge shouldn't have any large protrusions (knobs) or intrusions (holes)
        # However, the normal direction might point inwards or outwards.
        # We just check the amplitude of the deviation: max deviation.
        deviation = max_dist_outward - max_dist_inward
        
        if points_on_this_side and deviation < edge_length * 0.15:
            straight_edges += 1
            
    if straight_edges >= 2: return 'corner'
    elif straight_edges == 1: return 'edge'
    return 'interior'

img = load_image('puzzle_solver/data/pieces.jpg')
d = PieceDetector()
pieces = d.detect_pieces(img)

counts = {'interior': 0, 'edge': 0, 'corner': 0}
for idx, p in enumerate(pieces):
    cat = classify_piece_type(p['contour'])
    counts[cat] += 1
print(counts)
