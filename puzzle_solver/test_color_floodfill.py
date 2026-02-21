import cv2
import numpy as np

def test_color_floodfill():
    img = cv2.imread('data/pieces.jpg')
    # blur slightly to remove wood grain noise
    blurred = cv2.bilateralFilter(img, 9, 75, 75)
    
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # We flood fill from the 4 corners
    # Let's say tolerance is 20
    lo = (20, 20, 20)
    hi = (20, 20, 20)
    
    cv2.floodFill(blurred, mask, (0,0), (0,0,255), lo, hi, flags=(4 | (255 << 8)))
    cv2.floodFill(blurred, mask, (w-1,0), (0,0,255), lo, hi, flags=(4 | (255 << 8)))
    cv2.floodFill(blurred, mask, (0,h-1), (0,0,255), lo, hi, flags=(4 | (255 << 8)))
    cv2.floodFill(blurred, mask, (w-1,h-1), (0,0,255), lo, hi, flags=(4 | (255 << 8)))
    
    # The mask will have 255 where flooded
    flood_mask = mask[1:-1, 1:-1]
    
    # Invert to get the pieces
    pieces_mask = cv2.bitwise_not(flood_mask)
    
    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    pieces_mask = cv2.morphologyEx(pieces_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    pieces_mask = cv2.morphologyEx(pieces_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    contours, _ = cv2.findContours(pieces_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid = []
    for c in contours:
        a = cv2.contourArea(c)
        if a > 400:
            valid.append(a)
            
    print(f"Color FloodFill pieces: {len(valid)}")

if __name__ == '__main__':
    test_color_floodfill()
