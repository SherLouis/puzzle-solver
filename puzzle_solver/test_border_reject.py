import cv2
import numpy as np

def explore_pieces():
    image_path = 'data/pieces.jpg'
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 25, 100)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    areas = []
    
    for c in contours:
        area = cv2.contourArea(c)
        if area > 800:
            areas.append(area)
            
    areas.sort(reverse=True)
    print(f"Total pieces (>800): {len(areas)}")
    print("Smallest pieces:")
    print(areas[-10:])
    
if __name__ == '__main__':
    explore_pieces()
