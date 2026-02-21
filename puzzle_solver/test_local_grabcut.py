import cv2
import numpy as np

def test_local_grabcut():
    image_path = 'data/pieces.jpg'
    original_img = cv2.imread(image_path)
    
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edges_img = cv2.Canny(blurred_img, 25, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    crude_mask = cv2.morphologyEx(edges_img, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(crude_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidate_contours = [c for c in contours if cv2.contourArea(c) > 400]
    filtered_contours = []
    if candidate_contours:
        areas = [cv2.contourArea(c) for c in candidate_contours]
        median_area = np.median(areas)
        lower_bound = median_area * 0.5
        upper_bound = median_area * 1.5
        for c in candidate_contours:
            if lower_bound < cv2.contourArea(c) < upper_bound:
                filtered_contours.append(c)
                
    print(f"Crude robust pieces found: {len(filtered_contours)}")
    
    # 2. Localized Refinement with GrabCut
    refined_pieces_img = np.zeros_like(original_img)
    refined_count = 0
    m_pad = 10
    
    for i, crude_cnt in enumerate(filtered_contours):
        x, y, w, h = cv2.boundingRect(crude_cnt)
        x_start = max(0, x - m_pad)
        y_start = max(0, y - m_pad)
        x_end = min(original_img.shape[1], x + w + m_pad)
        y_end = min(original_img.shape[0], y + h + m_pad)
        
        patch_w = x_end - x_start
        patch_h = y_end - y_start
        
        patch_rgb = original_img[y_start:y_end, x_start:x_end]
        
        # Setup mask for GrabCut
        mask = np.zeros((patch_h, patch_w), np.uint8)
        
        # Create models
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)
        
        # Define rectangle containing the piece
        # We leave 2 pixels on the border as sure background
        rect = (2, 2, patch_w - 4, patch_h - 4)
        
        # Let's run GrabCut strictly initialized with rect
        cv2.grabCut(patch_rgb, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        
        mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
        
        pc_cnts, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if pc_cnts:
            # Add patch offset back
            exact_cnt = max(pc_cnts, key=cv2.contourArea) + [x_start, y_start]
            cv2.drawContours(refined_pieces_img, [exact_cnt], -1, (0, 255, 0), 2)
            refined_count += 1
            
    print(f"Refined {refined_count} pieces dynamically with GrabCut!")
    cv2.imwrite('debug_local_grabcut.jpg', refined_pieces_img)

if __name__ == '__main__':
    test_local_grabcut()
