import cv2
import numpy as np
import matplotlib.pyplot as plt
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

ratios = []
for idx, p in enumerate(pieces):
    r = get_compactness(p['contour'])
    ratios.append(r)
    
ratios.sort()
for i, r in enumerate(ratios):
    print(f"Piece sorted {i}: ratio = {r:.3f}")
