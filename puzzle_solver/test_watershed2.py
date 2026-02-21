import cv2
import numpy as np

def test_watershed2():
    image_path = 'data/pieces.jpg'
    img = cv2.imread(image_path)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # 1. Background Model from Border
    h, w = lab.shape[:2]
    bw = 5
    border_pixels = []
    border_pixels.append(lab[0:bw, :].reshape(-1, 3))
    border_pixels.append(lab[-bw:, :].reshape(-1, 3))
    border_pixels.append(lab[:, 0:bw].reshape(-1, 3))
    border_pixels.append(lab[:, -bw:].reshape(-1, 3))
    bg_samples = np.vstack(border_pixels).astype(np.float64)
    bg_mean = np.median(bg_samples, axis=0)
    
    # distance from bg
    diff = lab.astype(np.float64) - bg_mean
    dist = np.linalg.norm(diff, axis=2)
    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 2. Extract SURE_FG
    # High threshold ensures we only get the absolute core of the distinct pieces
    ret, sure_fg = cv2.threshold(dist, 50, 255, cv2.THRESH_BINARY)
    
    # Morphological clean up to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = np.zeros_like(sure_fg)
    
    # 3. Supposing the edge of the photo is part of the Background
    sure_bg[0:bw, :] = 255
    sure_bg[-bw:, :] = 255
    sure_bg[:, 0:bw] = 255
    sure_bg[:, -bw:] = 255
    
    # 4. Markers creation
    ret_comp, markers = cv2.connectedComponents(sure_fg)
    
    markers = markers + 1 # shift foreground by 1
    markers[sure_bg == 255] = 1 # Background is 1
    markers[sure_fg == 0] = 0 # Unknown is 0
    # Add border again just in case
    markers[0:bw, :] = 1
    markers[-bw:, :] = 1
    markers[:, 0:bw] = 1
    markers[:, -bw:] = 1
    
    # 5. Watershed
    # We blur the image slightly for smoother watershed boundaries
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    cv2.watershed(img_blur, markers)
    
    # 6. Count valid
    valid_pieces = 0
    for i in range(2, ret_comp + 1):
        mask = (markers == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            a = cv2.contourArea(contours[0])
            if a > 400:
                valid_pieces += 1
                
    print(f"Watershed combined pieces: {valid_pieces}")

if __name__ == '__main__':
    test_watershed2()
