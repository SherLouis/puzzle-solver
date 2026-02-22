import cv2
import numpy as np

class PieceDetector:
    def __init__(self, min_area=400, max_area=100000):
        self.min_area = min_area
        self.max_area = max_area

    def detect_pieces(self, image):
        """
        Detects puzzle pieces in the image using a robust Two-Pass Localized Strategy.
        Returns a list of dictionaries, each containing:
        - 'contour': The exact grabbed contour of the piece
        - 'bbox': Bounding box (x, y, w, h)
        - 'image': Cropped image of the piece
        - 'image_alpha': Cropped image of the piece with transparency
        - 'mask': Binary localized mask of the piece
        - 'center': (cx, cy) center of the piece in original image
        """
        # Pass 1: Global Crude Detection
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
        
        edges_img = cv2.Canny(blurred_img, 25, 100)
        
        # Aggressively seal edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        crude_mask = cv2.morphologyEx(edges_img, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(crude_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Pass 2: Statistical Outlier Rejection
        candidate_contours = []
        candidate_areas = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.min_area:
                candidate_contours.append(cnt)
                candidate_areas.append(area)
                
        if not candidate_areas:
             return []
             
        median_area = np.median(candidate_areas)
        lower_bound = median_area * 0.5
        upper_bound = median_area * 1.5
        
        filtered_crude_contours = []
        for cnt in candidate_contours:
            if lower_bound < cv2.contourArea(cnt) < upper_bound:
                filtered_crude_contours.append(cnt)
                
        # Pass 3: Dynamically Localized Piece Refinement (GrabCut)
        pieces = []
        h_full, w_full = image.shape[:2]
        m_pad = 10 
        
        for crude_cnt in filtered_crude_contours:
            x, y, w, h = cv2.boundingRect(crude_cnt)
            
            x_start = max(0, x - m_pad)
            y_start = max(0, y - m_pad)
            x_end = min(w_full, x + w + m_pad)
            y_end = min(h_full, y + h + m_pad)
            
            patch_w = x_end - x_start
            patch_h = y_end - y_start
            
            patch_rgb = image[y_start:y_end, x_start:x_end]
            
            mask = np.zeros((patch_h, patch_w), np.uint8)
            bgdModel = np.zeros((1,65), np.float64)
            fgdModel = np.zeros((1,65), np.float64)
            
            # 2 pixels border as sure background
            rect = (2, 2, patch_w - 4, patch_h - 4)
            
            # Only run GrabCut if rect is valid
            if patch_w <= 4 or patch_h <= 4:
                continue
                
            cv2.grabCut(patch_rgb, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            grabcut_mask = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
            
            pc_cnts, _ = cv2.findContours(grabcut_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if pc_cnts:
                exact_cnt_local = max(pc_cnts, key=cv2.contourArea)
                
                # Check area again to avoid weird grabcut artifacts
                if cv2.contourArea(exact_cnt_local) < self.min_area:
                    continue
                    
                exact_cnt_global = exact_cnt_local + [x_start, y_start]
                pieces.append(self._extract_refined_piece(image, exact_cnt_global, grabcut_mask, x_start, y_start, patch_w, patch_h))
                
        return pieces

    def _extract_refined_piece(self, image, global_contour, local_mask, x_start, y_start, patch_w, patch_h):
        # Cropped image from the exact bounding box of the final local contour
        lx, ly, lw, lh = cv2.boundingRect(global_contour - [x_start, y_start])
        
        # Original piece image segment securely bounds
        piece_img = image[y_start + ly : y_start + ly + lh, x_start + lx : x_start + lx + lw].copy()
        
        local_mask_alpha = local_mask * 255
        piece_mask = local_mask_alpha[ly : ly + lh, lx : lx + lw].copy()
        
        b, g, r = cv2.split(piece_img)
        rgba = [b, g, r, piece_mask]
        piece_with_alpha = cv2.merge(rgba, 4)
        
        gx, gy, gw, gh = cv2.boundingRect(global_contour)
        
        M = cv2.moments(global_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = gx + gw // 2, gy + gh // 2
            
        return {
            'contour': global_contour,
            'bbox': (gx, gy, gw, gh),
            'image': piece_img,
            'image_alpha': piece_with_alpha,
            'mask': piece_mask,
            'center': (cX, cY)
        }
