import cv2
import numpy as np

def test_color_filter():
    image_path = 'data/pieces.jpg'
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 1. Structural extraction
    edges = cv2.Canny(blurred, 25, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract size-based candidates
    candidates = [c for c in contours if cv2.contourArea(c) > 800]
    
    # 2. Extract strictly known background color from the edges
    h, w = img.shape[:2]
    bw = 5
    border_pixels = []
    border_pixels.append(img[0:bw, :].reshape(-1, 3))
    border_pixels.append(img[-bw:, :].reshape(-1, 3))
    border_pixels.append(img[:, 0:bw].reshape(-1, 3))
    border_pixels.append(img[:, -bw:].reshape(-1, 3))
    
    bg_samples = np.vstack(border_pixels).astype(np.float64)
    # Median is highly robust to noise on the border
    bg_color_bgr = np.median(bg_samples, axis=0) 
    
    # Convert bg color to LAB for better perceptual Euclidean distance
    # np.uint8 requires 3D array for cvtColor
    bg_color_3d = np.uint8([[[bg_color_bgr[0], bg_color_bgr[1], bg_color_bgr[2]]]])
    bg_color_lab = cv2.cvtColor(bg_color_3d, cv2.COLOR_BGR2LAB)[0][0]
    
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    valid_pieces = []
    
    # 3. Filter candidates by confirming they are distinctly NOT the background color
    for cnt in candidates:
        # Create mask for this piece
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
        
        # Calculate mean color in LAB space
        mean_lab = cv2.mean(img_lab, mask=mask)[:3]
        
        # Euclidean perceptual distance
        curr_dist = np.linalg.norm(np.array(mean_lab) - np.array(bg_color_lab))
        
        if curr_dist > 5.0: # If it's visually distinct from the table
            valid_pieces.append(cnt)
        else:
            print(f"Rejected a false piece! Area: {cv2.contourArea(cnt)}, Dist: {curr_dist:.2f}")
            
    print(f"Total pieces found after color validation: {len(valid_pieces)}")

if __name__ == '__main__':
    test_color_filter()
