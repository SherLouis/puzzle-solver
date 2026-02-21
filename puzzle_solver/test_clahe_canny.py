import cv2
import numpy as np

def test_clahe_canny():
    image_path = 'data/pieces.jpg'
    img = cv2.imread(image_path)
    
    # 1. Convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Add CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(gray)
    
    # 3. Blur
    blurred = cv2.GaussianBlur(cl1, (5, 5), 0)
    
    # 4. Canny
    for low in [20, 30, 40, 50, 60, 80]:
        for high in [80, 100, 120, 150, 200]:
            edges = cv2.Canny(blurred, low, high)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
            
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid = [c for c in contours if 2000 < cv2.contourArea(c) < 100000]
            if len(valid) >= 48:
                print(f"CLAHE+Canny ({low}, {high}): {len(valid)} pieces")

if __name__ == '__main__':
    test_clahe_canny()
