import cv2
import numpy as np

def test_local_watershed():
    image_path = 'data/pieces.jpg'
    original_img = cv2.imread(image_path)
    
    # 1. Initial crude segmentation (like the robust notebook)
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    
    # Canny Edge Detection
    edges_img = cv2.Canny(blurred_img, 25, 100)
    
    # Crude Morphology just to locate the pieces
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    crude_mask = cv2.morphologyEx(edges_img, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(crude_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidate_contours = [c for c in contours if cv2.contourArea(c) > 400]
    
    # Statistical filtering (like our robust notebook)
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
    
    # 2. Localized Refinement!
    # By analyzing a tight bounding box around each piece, we avoid boundary distortion
    # and we don't rely on the global background properties.
    refined_pieces_img = np.zeros_like(original_img)
    refined_count = 0
    
    m_pad = 15 # generous margin ensuring we hit background
    
    # Pre-blur for watershed
    watershed_source = cv2.bilateralFilter(original_img, 9, 75, 75)
    
    for i, crude_cnt in enumerate(filtered_contours):
        x, y, w, h = cv2.boundingRect(crude_cnt)
        
        # Add a margin to guarantee we include background around the piece
        x_start = max(0, x - m_pad)
        y_start = max(0, y - m_pad)
        x_end = min(original_img.shape[1], x + w + m_pad)
        y_end = min(original_img.shape[0], y + h + m_pad)
        
        patch_w = x_end - x_start
        patch_h = y_end - y_start
        
        patch_rgb = watershed_source[y_start:y_end, x_start:x_end]
        
        # Setup markers for this tiny patch
        markers = np.zeros((patch_h, patch_w), dtype=np.int32)
        
        # Draw Background: the outer perimeter of the patch
        # thickness=2 to ensure a thick enough bg seed
        cv2.rectangle(markers, (0, 0), (patch_w - 1, patch_h - 1), 1, thickness=5)
        
        # Draw Foreground: The core of the crude contour
        # Instead of just the center point (which might accidentally hit a hole),
        # we draw the crude contour shifted to patch coordinates, and erode it heavily!
        crude_patch_mask = np.zeros((patch_h, patch_w), dtype=np.uint8)
        shifted_cnt = crude_cnt - [x_start, y_start]
        cv2.drawContours(crude_patch_mask, [shifted_cnt], -1, 255, thickness=cv2.FILLED)
        
        # Erode the crude area to ensure we ONLY flag the "sure core" of the piece as foreground
        core_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)) # Shrink deeply
        sure_fg = cv2.erode(crude_patch_mask, core_kernel, iterations=1)
        
        markers[sure_fg == 255] = 2 # Foreground marker is 2
        
        # Run Localized Watershed
        cv2.watershed(patch_rgb, markers)
        
        # Mask where markers == 2
        piece_patch_mask = (markers == 2).astype(np.uint8) * 255
        
        # Find exactly the contour of the local piece mask
        pc_cnts, _ = cv2.findContours(piece_patch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if pc_cnts:
            # Add patch offset back to the global coordinates
            exact_cnt = pc_cnts[0] + [x_start, y_start]
            cv2.drawContours(refined_pieces_img, [exact_cnt], -1, (0, 255, 0), 2)
            refined_count += 1

    print(f"Refined {refined_count} pieces dynamically!")
    
    cv2.imwrite('debug_local_watershed.jpg', refined_pieces_img)

if __name__ == '__main__':
    test_local_watershed()
