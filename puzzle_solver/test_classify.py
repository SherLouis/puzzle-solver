import cv2
import numpy as np
from src.image_processing import load_image
from src.segmentation import PieceDetector

def classify_piece(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int64(box) # changed from np.int0
    
    # Calculate lengths of the 4 sides of the minAreaRect
    sides = []
    for i in range(4):
        p1 = box[i]
        p2 = box[(i+1)%4]
        length = np.linalg.norm(p1 - p2)
        sides.append((p1, p2, length))
        
    straight_edges = 0
    
    # For each side, check how many contour points are close to it
    for i in range(4):
        p1 = box[i]
        p2 = box[(i+1)%4]
        length = sides[i][2]
        if length < 10: # Too short to be a valid side
            continue
            
        dy = p2[1] - p1[1]
        dx = p2[0] - p1[0]
        c = p2[0]*p1[1] - p2[1]*p1[0]
        
        close_points = 0
        for pt in contour:
            x0, y0 = pt[0]
            dist = abs(dy*x0 - dx*y0 + c) / length
            if dist < 5.0:  # 5 pixels tolerance
                close_points += 1
                
        # If the number of points close to the line is roughly equal to the length
        # (meaning it's a straight line along the contour)
        if close_points > length * 0.4:  # At least 40% of the side length is flat
            straight_edges += 1
            
    if straight_edges == 0:
        return 'interior'
    elif straight_edges == 1:
        return 'edge'
    elif straight_edges == 2:
        return 'corner'
    else:
        # Sometimes 3 sides might trigger if the piece has very straight knobs/holes, 
        # but 2+ is generally corner.
        return 'corner'

try:
    pieces_image = load_image('puzzle_solver/data/pieces.jpg')
except:
    pieces_image = load_image('data/pieces.jpg')

detector = PieceDetector()
pieces = detector.detect_pieces(pieces_image)
print(f"Found {len(pieces)} pieces.")

counts = {'interior': 0, 'edge': 0, 'corner': 0}
for idx, p in enumerate(pieces):
    category = classify_piece(p['contour'])
    counts[category] += 1
    print(f"Piece {idx}: {category}")

print(counts)
