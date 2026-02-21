import cv2
import numpy as np

def test_illumination():
    image_path = 'data/pieces.jpg'
    original_img = cv2.imread(image_path)
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    # 1. CLAHE to normalize lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(gray)
    
    # 2. Estimate background
    # Top-hat transform can also work. Let's try cv2.adaptiveThreshold
    # on original or clahe.
    
    for bs in [51, 101, 151, 201]:
        for C in [-5, 0, 5, 10, 15]:
            thresh = cv2.adaptiveThreshold(cl1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, bs, C)
                                           
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)
            
            contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_contours = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 2000 < area < 100000:
                    valid_contours.append(area)
            
            if len(valid_contours) >= 45:
                print(f"Adaptive (bs={bs}, C={C}): Found {len(valid_contours)} pieces")

if __name__ == '__main__':
    test_illumination()
