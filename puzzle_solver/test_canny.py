import cv2
import numpy as np

def explore_canny_deep():
    image_path = 'data/pieces.jpg'
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Let's use automatic Canny thresholding using median
    v = np.median(blurred)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(blurred, lower, upper)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Optional: Fill holes using findContours & drawContours
    # Because sometimes pieces are not completely empty inside
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 2000 < area < 100000:
            valid_contours.append(cnt)
    
    print(f"Canny automatic (lower={lower}, upper={upper}): Found {len(valid_contours)} pieces")
    
    display = img.copy()
    cv2.drawContours(display, valid_contours, -1, (0, 0, 255), 3)
    for i, c in enumerate(valid_contours):
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(display, str(i), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    cv2.imwrite('debug_canny_auto.jpg', display)

if __name__ == '__main__':
    explore_canny_deep()
