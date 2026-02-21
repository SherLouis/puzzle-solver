import cv2
import numpy as np

def test_watershed():
    img = cv2.imread('data/pieces.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Basic thresholding to get a probable mask
    # We can use Adaptive Threshold
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 51, 10)
                                   
    # Noise removal
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # Closing to fill small holes inside pieces
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 2. Sure Foreground (SURE_FG) - by shrinking the pieces
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
    # pieces are usually thick. thresholding distance transform gives their cores.
    # a piece is typically ~40 pixels wide. Let's threshold at 10 pixels depth.
    ret, sure_fg = cv2.threshold(dist_transform, 10, 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # 3. Create Markers
    markers = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)
    
    # 4. Connected Components on SURE_FG to get distinct pieces
    ret, markers_fg = cv2.connectedComponents(sure_fg)
    
    # Shift foreground markers by 1 so background can be 1
    # Actually wait: 0 means unknown.
    # Let's set the SURE_FG
    markers = markers_fg + 1
    
    # 5. Supposing the edge of the photo is part of the BACKGROUND
    # We set the perimeter pixels to 1 (Background class)
    bw = 10 # border width
    markers[0:bw, :] = 1
    markers[-bw:, :] = 1
    markers[:, 0:bw] = 1
    markers[:, -bw:] = 1
    
    # Unknown regions are currently where markers == 1 (from shift)
    # Let's rebuild properly:
    # 0 = Unknown
    markers = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)
    
    # Add FG (starting from 2)
    # markers_fg has 0 for bg, 1..N for fg components. 
    # We add 1 so they are 2..N+1
    markers[markers_fg > 0] = markers_fg[markers_fg > 0] + 1
    
    # Add BG border
    markers[0:bw, :] = 1
    markers[-bw:, :] = 1
    markers[:, 0:bw] = 1
    markers[:, -bw:] = 1
    
    # 6. Apply Watershed
    cv2.watershed(img, markers)
    
    # markers now contains 1 for bg, 2..N+1 for pieces, -1 for boundaries
    
    # Let's count valid pieces!
    valid_pieces = 0
    areas = []
    
    # ret is number of components from connectedComponents (includes 0 as bg)
    for i in range(2, ret + 1):
        mask = (markers == i).astype(np.uint8) * 255
        # find contour for this mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            a = cv2.contourArea(contours[0])
            if a > 400: # Filter noise
                valid_pieces += 1
                areas.append(int(a))
                
    print(f"Watershed pieces: {valid_pieces}")
    print(f"Top 5 areas: {sorted(areas, reverse=True)[:5]}")

if __name__ == '__main__':
    test_watershed()
