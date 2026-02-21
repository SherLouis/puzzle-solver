import cv2
import numpy as np

def test_watershed_50():
    image_path = 'data/pieces.jpg'
    original_img = cv2.imread(image_path)
    
    # 1. Initial crude segmentation (like the robust notebook)
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    
    # Canny Edge Detection
    edges_img = cv2.Canny(blurred_img, 25, 100)
    
    # Crude Morphology (we accept shape distortion here just to locate the pieces)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    crude_mask = cv2.morphologyEx(edges_img, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find crude contours
    contours, _ = cv2.findContours(crude_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter crude pieces (Area bounds)
    # We found earlier that standard pieces are ~800+ area
    valid_contours = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 800:
            valid_contours.append(cnt)
            
    print(f"Initial crude pieces found: {len(valid_contours)}")
    
    # 2. Setup Watershed Markers
    h, w = original_img.shape[:2]
    markers = np.zeros((h, w), dtype=np.int32)
    
    # Draw Background. We know background is everything OUTSIDE these crude contours.
    # We want to be careful and not label piece borders as background, so we dilate the crude contours
    # and invert.
    crude_filled = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(crude_filled, valid_contours, -1, 255, thickness=cv2.FILLED)
    
    sure_bg = cv2.dilate(crude_filled, kernel, iterations=3)
    sure_bg = cv2.bitwise_not(sure_bg)
    markers[sure_bg == 255] = 1 # 1 is Background
    
    # Draw foreground markers (the piece centers!)
    for i, cnt in enumerate(valid_contours):
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Draw a solid circle safely inside the piece
            cv2.circle(markers, (cX, cY), 5, i + 2, -1)
            
    # Apply Watershed on the RGB image (we can blur it slightly to remove wood grain noise)
    watershed_img = cv2.bilateralFilter(original_img, 9, 75, 75)
    cv2.watershed(watershed_img, markers)
    
    # Verify exact piece counts
    watershed_pieces = []
    for i in range(2, len(valid_contours) + 2):
        mask = (markers == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            # largest contour for this marker
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) > 400:
                watershed_pieces.append(c)
                
    print(f"Watershed refined pieces EXACT: {len(watershed_pieces)}")
    
    disp = original_img.copy()
    cv2.drawContours(disp, watershed_pieces, -1, (0, 255, 0), 2)
    cv2.imwrite('debug_watershed_50.jpg', disp)

if __name__ == '__main__':
    test_watershed_50()
