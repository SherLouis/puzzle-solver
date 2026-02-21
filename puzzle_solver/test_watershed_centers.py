import cv2
import numpy as np

def test_watershed_centers():
    image_path = 'data/pieces.jpg'
    img = cv2.imread(image_path)
    blurred = cv2.bilateralFilter(img, 9, 75, 75)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    # 1. Initial crude segmentation (Flood Fill from borders)
    h, w = img.shape[:2]
    flood_mask = np.zeros((h+2, w+2), np.uint8)
    tol = (15, 15, 15)
    flags = 4 | (255 << 8)
    tmp = blurred.copy()
    
    cv2.floodFill(tmp, flood_mask, (0,0), (0,0,255), tol, tol, flags)
    cv2.floodFill(tmp, flood_mask, (w-1,0), (0,0,255), tol, tol, flags)
    cv2.floodFill(tmp, flood_mask, (0,h-1), (0,0,255), tol, tol, flags)
    cv2.floodFill(tmp, flood_mask, (w-1,h-1), (0,0,255), tol, tol, flags)
    
    bg_mask = flood_mask[1:-1, 1:-1]
    crude_pieces = cv2.bitwise_not(bg_mask)
    
    # Clean up crude pieces slightly just to get valid connected components
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    crude_pieces = cv2.morphologyEx(crude_pieces, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 2. Filter components by area to find true piece centers
    contours, _ = cv2.findContours(crude_pieces, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    for c in contours:
        a = cv2.contourArea(c)
        if 400 < a < 50000:
            valid_contours.append(c)
            
    print(f"Centers found: {len(valid_contours)}")
    
    # 3. Create Markers for Watershed
    markers = np.zeros((h, w), dtype=np.int32)
    
    # Let's define sure_bg correctly.
    # We want Watershed to find the EXACT edges without morphological distortion.
    # We'll use Canny edges as the "unknown" boundary zone.
    edges = cv2.Canny(gray, 20, 80)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    
    # Set sure_bg as the crude background minus the edge zones
    sure_bg = cv2.bitwise_and(bg_mask, cv2.bitwise_not(edges))
    markers[sure_bg == 255] = 1
    
    # 4. Draw piece centers as sure foreground
    for i, c in enumerate(valid_contours):
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Draw a solid circle at the center as the seed
            cv2.circle(markers, (cX, cY), 10, i + 2, -1)
            
    # Pixels where markers == 1 (bg) or >1 (fg) are seeds.
    # Pixels where markers == 0 are unknown boundaries that watershed will decide based on gradients.
    
    # Apply Watershed
    # Watershed needs the 8-bit 3-channel image. It heavily relies on the edges.
    img_for_ws = blurred.copy()
    cv2.watershed(img_for_ws, markers)
    
    # Markers now have the exact bounds.
    # -1 is boundary
    watershed_pieces = 0
    for i in range(2, len(valid_contours) + 2):
        mask = (markers == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            a = cv2.contourArea(cnts[0])
            if a > 400:
                watershed_pieces += 1
                
    print(f"Watershed refined pieces: {watershed_pieces}")
    
    # Display the boundaries
    img[markers == -1] = [0, 0, 255] # Mark boundaries in red
    cv2.imwrite('debug_watershed.jpg', img)

if __name__ == '__main__':
    test_watershed_centers()
