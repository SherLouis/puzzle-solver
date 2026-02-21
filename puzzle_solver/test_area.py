import cv2
import numpy as np

def test_area():
    image_path = 'data/pieces.jpg'
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    all_areas = sorted([int(cv2.contourArea(c)) for c in contours], reverse=True)
    print("Top 60 areas:", all_areas[:60])
    
    # Let's also check without Canny, using Adaptive again with Threshold 99 -> Area might be small!
    
if __name__ == '__main__':
    test_area()
