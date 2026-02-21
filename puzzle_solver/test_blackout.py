import cv2
import numpy as np

def test_blackout():
    image_path = 'data/pieces.jpg'
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 1. Suppose the edge of the photo is background
    # Set the outer 10 pixels to a solid flat color (median of the border)
    # This destroys any Canny noise on the edges
    bw = 10
    med = np.median(blurred[0:bw, :])
    blurred[0:bw, :] = med
    blurred[-bw:, :] = med
    blurred[:, 0:bw] = med
    blurred[:, -bw:] = med
    
    edges = cv2.Canny(blurred, 25, 100)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid = []
    
    for c in contours:
        area = cv2.contourArea(c)
        if area > 800:
            valid.append(area)
            
    print(f"Pieces found: {len(valid)}")

if __name__ == '__main__':
    test_blackout()
