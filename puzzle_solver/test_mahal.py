import cv2
import numpy as np
from scipy.spatial.distance import cdist

def test_bg_model():
    img = cv2.imread('data/pieces.jpg')
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    
    # 1. Get border pixels
    h, w = img.shape[:2]
    bw = 10 # border width
    border_pixels = []
    border_pixels.append(blurred[0:bw, :].reshape(-1, 3))
    border_pixels.append(blurred[-bw:, :].reshape(-1, 3))
    border_pixels.append(blurred[:, 0:bw].reshape(-1, 3))
    border_pixels.append(blurred[:, -bw:].reshape(-1, 3))
    
    bg_samples = np.vstack(border_pixels).astype(np.float64)
    
    # 2. Compute Mean & Covariance
    mean = np.mean(bg_samples, axis=0)
    cov = np.cov(bg_samples.T)
    inv_cov = np.linalg.inv(cov)
    
    # 3. Compute Mahalanobis distance for all pixels
    blurred_flat = blurred.reshape(-1, 3).astype(np.float64)
    diff = blurred_flat - mean
    
    # We can do this efficiently:
    # dist = sum( (diff * inv_cov) * diff ) along axis=1
    left = np.dot(diff, inv_cov)
    dist = np.sum(left * diff, axis=1)
    
    dist_img = dist.reshape(h, w)
    
    # 4. Threshold distance
    # A pixel belongs to background if its distance is small (e.g., < threshold)
    mask = (dist_img > 15).astype(np.uint8) * 255
    
    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid = []
    for c in contours:
        a = cv2.contourArea(c)
        if a > 400:
            valid.append(a)
            
    print(f"Mahalanobis pieces: {len(valid)}")

if __name__ == '__main__':
    test_bg_model()
