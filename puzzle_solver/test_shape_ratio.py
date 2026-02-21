import cv2
import numpy as np

def test_shape_ratio():
    image_path = 'data/pieces.jpg'
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv2.Canny(blurred, 25, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid = []
    
    for c in contours:
        area = cv2.contourArea(c)
        if area > 800:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h if h != 0 else 0
            
            # Pieces are generally squarish, maybe 0.5 to 2.0
            if 0.5 < aspect_ratio < 2.0:
                valid.append(area)
            else:
                print(f"Rejected piece! Area: {area}, Aspect Ratio: {aspect_ratio:.2f}")
                
    print(f"Total pieces found: {len(valid)}")

if __name__ == '__main__':
    test_shape_ratio()
