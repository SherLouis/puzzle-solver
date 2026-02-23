import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.image_processing import load_image
from src.segmentation import PieceDetector

def get_piece_edges(contour):
    # Find convex hull
    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)
    
    # Actually, minAreaRect is better, but maybe with tighter tolerance and only counting contiguous points
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int64(box)
    
    straight_edges = 0
    
    for i in range(4):
        p1 = box[i]
        p2 = box[(i+1)%4]
        length = np.linalg.norm(p1 - p2)
        if length < 20: continue
            
        dy = p2[1] - p1[1]
        dx = p2[0] - p1[0]
        c = p2[0]*p1[1] - p2[1]*p1[0]
        
        # calculate max contiguous points close to this line
        max_contiguous = 0
        current_contiguous = 0
        
        for pt in contour:
            x0, y0 = pt[0]
            dist = abs(dy*x0 - dx*y0 + c) / length
            if dist < 8.0:
                current_contiguous += 1
                if current_contiguous > max_contiguous:
                    max_contiguous = current_contiguous
            else:
                current_contiguous = 0
                
        # Handle wrap around
        current_contiguous = 0
        for pt in reversed(contour):
            x0, y0 = pt[0]
            dist = abs(dy*x0 - dx*y0 + c) / length
            if dist < 8.0:
                current_contiguous += 1
                if current_contiguous + max_contiguous > len(contour):
                    break
            else:
                break
        max_contiguous += current_contiguous
        
        # If the continuous straight segment is at least 30% of the box side
        if max_contiguous > length * 0.4:
            straight_edges += 1
            
    if straight_edges >= 2:
        return 'corner'
    elif straight_edges == 1:
        return 'edge'
    else:
        return 'interior'

img = load_image('puzzle_solver/data/pieces.jpg')
d = PieceDetector()
pieces = d.detect_pieces(img)

counts = {'interior': 0, 'edge': 0, 'corner': 0}
for p in pieces:
    cat = get_piece_edges(p['contour'])
    counts[cat] += 1
print(counts)
