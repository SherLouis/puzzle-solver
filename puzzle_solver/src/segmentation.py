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
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding to handle lighting variations
        # Inverting because pieces are usually lighter than the gaps, 
        # but here we rely on the edge difference. 
        # Using Canny might be better for "pieces on table" if contrast is high.
        # Let's try adaptive threshold first as it's robust.
        
        # Preprocessing
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Thresholding
        # Otsu's thresholding often works better for "objects on background" 
        # where there's a bi-modal histogram.
        # We try both INV and normal, assuming pieces are brighter or distinct.
        # If the table is dark and pieces are bright -> THRESH_BINARY
        # If table is light and pieces are dark -> THRESH_BINARY_INV
        # Here we assume Table is the background (usually dominant peak).
        
        # Adaptive Thresholding with block size 51 worked best in testing, finding ~27 pieces.
        # Pieces are likely multicolored, so adaptive helps.
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 51, 2)
        
        # thresh_val, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Remove auto-flip for now
        # if cv2.countNonZero(thresh) > (image.shape[0] * image.shape[1] * 0.6):
        #      thresh = cv2.bitwise_not(thresh)

        # Morphological operations to remove noise and close gaps
        # Kernel size 3x3 matched the visualization script success.
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        pieces = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Extract the piece with some padding
                pad = 5
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(image.shape[1], x + w + pad)
                y2 = min(image.shape[0], y + h + pad)
                
                piece_img = image[y1:y2, x1:x2].copy()
                
                # Create a mask for this specific piece
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                piece_mask = mask[y1:y2, x1:x2].copy()
                
                # Apply mask to alpha channel for transparency (optional, good for viz)
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

                pieces.append({
                    'contour': contour,
                    'bbox': (x, y, w, h),
                    'image': piece_img, # BGR
                    'image_alpha': piece_with_alpha, # BGRA
                    'mask': piece_mask,
                    'center': (cX, cY)
                })
        
        return pieces
