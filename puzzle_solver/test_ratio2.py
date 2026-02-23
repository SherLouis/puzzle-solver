import cv2
import numpy as np
import os
from src.image_processing import load_image
from src.segmentation import PieceDetector

def get_compactness(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if area == 0: return 0
    return perimeter / np.sqrt(area)

img = load_image('puzzle_solver/data/pieces.jpg')
d = PieceDetector()
pieces = d.detect_pieces(img)

# Ensure output directory exists
os.makedirs('puzzle_solver/debug_pieces', exist_ok=True)

classified_counts = {'corner': 0, 'edge': 0, 'interior': 0}

for idx, p in enumerate(pieces):
    r = get_compactness(p['contour'])
    if r < 5.8:
        cat = 'corner'
    elif r < 6.8: # We'll see if 6.8 is a good threshold for edges
        cat = 'edge'
    else:
        cat = 'interior'
        
    classified_counts[cat] += 1
    
    # Save image for verification
    cv2.imwrite(f'puzzle_solver/debug_pieces/{cat}_{idx}_ratio_{r:.2f}.png', p['image_alpha'])

print(f"Classification counts: {classified_counts}")
