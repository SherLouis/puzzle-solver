import cv2
import numpy as np
from .utils import resize_image

class PieceDetector:
    def __init__(self, min_area=500, max_area=100000):
        self.min_area = min_area
        self.max_area = max_area

    def detect_pieces(self, image):
        """
        Detects puzzle pieces in the image.
        Returns a list of dictionaries, each containing:
        - 'contour': The contour of the piece
        - 'bbox': Bounding box (x, y, w, h)
        - 'image': Cropped image of the piece (with transparency/mask)
        - 'mask': Binary mask of the piece
        - 'center': (cx, cy) center of the piece in original image
        """
        # Preprocessing
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # 1. Background Color Removal (Wood)
        # Based on optimization (DarkWood config): 
        # Captures wood texture better including dark grain
        lower_wood = np.array([5, 40, 20])    
        upper_wood = np.array([30, 255, 180])
        
        # Create mask of the wood table
        bg_mask = cv2.inRange(hsv, lower_wood, upper_wood)
        
        # Invert to get pieces (foreground)
        fg_mask = cv2.bitwise_not(bg_mask)
        
        # 2. Refine Foreground
        # Remove small noise
        kernel_small = np.ones((3,3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
        
        # Aggressive closing to fill holes in pieces
        kernel_large = np.ones((7,7), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_large, iterations=4)
        
        # 3. Initial Contour Detection
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = []
        areas = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                valid_contours.append(contour)
                areas.append(area)
        
        if not areas:
            return []
            
        pieces = []
        for contour in valid_contours:
            # Just extract the piece directly without splitting
            pieces.append(self._extract_piece(image, contour))
        
        return pieces

    def _extract_piece(self, image, contour):
        x, y, w, h = cv2.boundingRect(contour)
        
        # Extract the piece with some padding
        pad = 5
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)
        
        piece_img = image[y1:y2, x1:x2].copy()
        
        # Create a clean mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        piece_mask = mask[y1:y2, x1:x2].copy()
        
        # Apply mask to alpha channel
        b, g, r = cv2.split(piece_img)
        rgba = [b, g, r, piece_mask]
        piece_with_alpha = cv2.merge(rgba, 4)

        # Calculate center
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = x + w // 2, y + h // 2

        return {
            'contour': contour,
            'bbox': (x, y, w, h),
            'image': piece_img,
            'image_alpha': piece_with_alpha,
            'mask': piece_mask,
            'center': (cX, cY)
        }
