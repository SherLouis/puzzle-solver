import cv2
import numpy as np

def test_grabcut():
    img = cv2.imread('data/pieces.jpg')
    mask = np.zeros(img.shape[:2], np.uint8)
    
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    
    # Define a rectangle leaving a 10px border
    h, w = img.shape[:2]
    rect = (10, 10, w-20, h-20)
    
    print("Running GrabCut... This might take a few seconds.")
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    
    # Mask is 0 (sure bg), 1 (sure fg), 2 (probable bg), 3 (probable fg)
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    
    # Closing to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid = []
    for c in contours:
        a = cv2.contourArea(c)
        if a > 400:
            valid.append(a)
            
    print(f"GrabCut pieces: {len(valid)}")

if __name__ == '__main__':
    test_grabcut()
