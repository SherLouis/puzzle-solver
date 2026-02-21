import cv2
import numpy as np

def test_flood_sweep():
    img = cv2.imread('data/pieces.jpg')
    blurred = cv2.bilateralFilter(img, 9, 75, 75)
    
    h, w = img.shape[:2]
    
    for tol in [10, 20, 30, 40, 50, 60, 70, 80]:
        mask = np.zeros((h+2, w+2), np.uint8)
        lo = (tol, tol, tol)
        hi = (tol, tol, tol)
        
        tmp = blurred.copy()
        cv2.floodFill(tmp, mask, (0,0), (0,0,255), lo, hi, flags=(4 | (255 << 8)))
        cv2.floodFill(tmp, mask, (w-1,0), (0,0,255), lo, hi, flags=(4 | (255 << 8)))
        cv2.floodFill(tmp, mask, (0,h-1), (0,0,255), lo, hi, flags=(4 | (255 << 8)))
        cv2.floodFill(tmp, mask, (w-1,h-1), (0,0,255), lo, hi, flags=(4 | (255 << 8)))
        
        flood_mask = mask[1:-1, 1:-1]
        pieces_mask = cv2.bitwise_not(flood_mask)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        pieces_mask = cv2.morphologyEx(pieces_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        pieces_mask = cv2.morphologyEx(pieces_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        contours, _ = cv2.findContours(pieces_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if 400 < cv2.contourArea(c) < 50000]
        
        print(f"Tol {tol}: {len(valid)} pieces")

if __name__ == '__main__':
    test_flood_sweep()
