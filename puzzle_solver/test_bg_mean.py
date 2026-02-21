import cv2
import numpy as np

def test_bg_mean():
    img = cv2.imread('data/pieces.jpg')
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    h, w = lab.shape[:2]
    bw = 5 # border width
    border_pixels = []
    border_pixels.append(lab[0:bw, :].reshape(-1, 3))
    border_pixels.append(lab[-bw:, :].reshape(-1, 3))
    border_pixels.append(lab[:, 0:bw].reshape(-1, 3))
    border_pixels.append(lab[:, -bw:].reshape(-1, 3))
    
    bg_samples = np.vstack(border_pixels).astype(np.float64)
    
    # Calculate median color of the border
    bg_median = np.median(bg_samples, axis=0)
    
    # Calculate distance to median color
    diff = lab.astype(np.float64) - bg_median
    dist = np.linalg.norm(diff, axis=2)
    
    # Normalize distance to 0-255
    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Threshold
    # distance > threshold means foreground piece
    # Try different thresholds
    for thr in [20, 30, 40, 50, 60]:
        mask = (dist > thr).astype(np.uint8) * 255
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if 400 < cv2.contourArea(c) < 50000]
        
        if len(valid) >= 45:
            print(f"Dist Threshold {thr}: found {len(valid)} pieces")

if __name__ == '__main__':
    test_bg_mean()
