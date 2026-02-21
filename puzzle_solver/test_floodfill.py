import cv2
import numpy as np

def test_flood_fill2():
    img = cv2.imread('data/pieces.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv2.Canny(blurred, 20, 80)
    
    # We aggressively close/dilate the edges so floodfill cannot leak into pieces
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    # also dilate
    closed = cv2.dilate(closed, kernel, iterations=2)
    
    h, w = closed.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    
    floodfilled = closed.copy()
    
    # Suppose the edge of the photo is background
    for i in range(w):
        if floodfilled[0, i] == 0:
            cv2.floodFill(floodfilled, mask, (i, 0), 255)
        if floodfilled[h-1, i] == 0:
            cv2.floodFill(floodfilled, mask, (i, h-1), 255)
            
    for j in range(h):
        if floodfilled[j, 0] == 0:
            cv2.floodFill(floodfilled, mask, (0, j), 255)
        if floodfilled[j, w-1] == 0:
            cv2.floodFill(floodfilled, mask, (w-1, j), 255)
    
    # background is now 255. Pieces are still 0 (if they were perfectly closed).
    pieces_mask = cv2.bitwise_not(floodfilled)
    
    # Now erode the pieces back to true size (since we dilated edges before)
    # pieces shrinked because edges grew inward.
    pieces_mask = cv2.dilate(pieces_mask, kernel, iterations=2)
    
    contours, _ = cv2.findContours(pieces_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid = []
    for c in contours:
        a = cv2.contourArea(c)
        if a > 400:
            valid.append(a)
            
    print(f"Dilated FloodFill pieces: {len(valid)}")

if __name__ == '__main__':
    test_flood_fill2()
