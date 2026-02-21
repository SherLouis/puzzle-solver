import cv2
import numpy as np

def test_backproject():
    img = cv2.imread('data/pieces.jpg')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    h, w = hsv.shape[:2]
    
    # 1. Create a mask for the border
    border_mask = np.zeros((h, w), np.uint8)
    border_thickness = 20
    border_mask[0:border_thickness, :] = 255
    border_mask[-border_thickness:, :] = 255
    border_mask[:, 0:border_thickness] = 255
    border_mask[:, -border_thickness:] = 255
    
    # 2. Calculate the Histogram of the Border Pixels in HSV
    # Using Hue and Saturation
    hist = cv2.calcHist([hsv], [0, 1], border_mask, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    
    # 3. BackProject the histogram onto the entire image
    backproj = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
    
    # Background has high probability (high values).
    # Objects have low probability (low values).
    
    # 4. Threshold
    ret, mask = cv2.threshold(backproj, 50, 255, cv2.THRESH_BINARY_INV)
    
    # 5. Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid = []
    for c in contours:
        a = cv2.contourArea(c)
        if a > 400:
            valid.append(a)
            
    print(f"BackProject pieces: {len(valid)}")

if __name__ == '__main__':
    test_backproject()
