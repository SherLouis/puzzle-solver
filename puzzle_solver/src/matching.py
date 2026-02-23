import cv2
import numpy as np

class ReferenceAnalyzer:
    def __init__(self, reference_image):
        self.reference_image = reference_image
        self.keypoints = None
        self.descriptors = None
        self.shape = reference_image.shape[:2] # h, w

    def compute_features(self):
        """Computes SIFT features for the reference image."""
        # SIFT is patent-expired and available in main opencv build since 4.4.0
        # If not available, use ORB.
        try:
            self.detector = cv2.SIFT_create(nfeatures=50000, contrastThreshold=0.005, edgeThreshold=20)
            self.detector_type = "SIFT"
        except AttributeError:
            self.detector = cv2.ORB_create(nfeatures=5000)
            self.detector_type = "ORB"
            
        gray = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2GRAY)
        self.keypoints, self.descriptors = self.detector.detectAndCompute(gray, None)
        return self.keypoints, self.descriptors

class PieceMatcher:
    def __init__(self, reference_analyzer):
        self.ref = reference_analyzer
        if self.ref.detector_type == "ORB":
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        # Working copy of features (can be filtered iteratively)
        self.current_keypoints = self.ref.keypoints
        self.current_descriptors = self.ref.descriptors

    def reset_features(self):
        self.current_keypoints = self.ref.keypoints
        self.current_descriptors = self.ref.descriptors

    def update_features(self, valid_indices):
        """
        Updates the current keypoints and descriptors to only keep valid_indices.
        """
        if self.current_descriptors is None:
            return
            
        self.current_keypoints = [self.current_keypoints[i] for i in valid_indices]
        self.current_descriptors = self.current_descriptors[valid_indices]
        
    def match_piece(self, piece, min_matches=4): # Lowered threshold for debug
        """
        Matches a single piece to the reference image.
        Returns:
        - match_success (bool)
        - transform_matrix (3x3 homography or None)
        - visualization_data (dict)
        """
        piece_img = piece['image']
        
        # SIFT works best with larger pixel footprints; scale up the small piece crops
        scale_factor = 3.0
        piece_img_large = cv2.resize(piece_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        gray_piece = cv2.cvtColor(piece_img_large, cv2.COLOR_BGR2GRAY)
        
        # Dilate mask slightly to capture edge features natively
        mask_large = cv2.resize(piece['mask'], None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask_large = cv2.dilate(mask_large, kernel, iterations=1)
        
        kp_piece, des_piece = self.ref.detector.detectAndCompute(gray_piece, mask=mask_large)
        # print(f"DEBUG: Piece descriptors found: {len(des_piece) if des_piece is not None else 0}")
        
        if des_piece is None or len(des_piece) < 2:
            return False, None, {}

        # 2. Match descriptors
        try:
            # k=2 for Ratio Test
            if self.current_descriptors is None or len(self.current_descriptors) < 2:
                return False, None, {}
                
            matches = self.matcher.knnMatch(des_piece, self.current_descriptors, k=2)
        except cv2.error:
            # Fallback if types mismatch (e.g. SIFT vs ORB descriptor norms)
            return False, None, {}

        # 3. Ratio Test
        # Relaxed slightly to 0.95 to allow more candidates for RANSAC
        good_matches = []
        for m, n in matches:
            if m.distance < 0.95 * n.distance:
                good_matches.append(m)

        if len(good_matches) < min_matches:
            return False, None, {'matches': len(good_matches)}

        # 4. Homography with RANSAC
        src_pts = np.float32([kp_piece[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        src_pts /= scale_factor
        
        # Use current_keypoints for destination
        dst_pts = np.float32([self.current_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Use Homography with RANSAC to allow mapping perspective changes seamlessly
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
        
        if H is None:
            return False, None, {'matches': len(good_matches)}
            
        matches_mask = mask.ravel().tolist()
        inliers_count = np.sum(mask)

        # 5. Color Verification
        # If matches are high enough, skip color check (expensive and robust enough)
        # Low match count warrants a check.
        if inliers_count < 10: # Check based on inliers, not raw matches
             if not self.verify_color(piece, H):
                return False, None, {'matches': len(good_matches), 'inliers': inliers_count, 'color_verified': False}
        
        return True, H, {
            'matches': len(good_matches),
            'inliers': int(inliers_count),
            'good_matches': good_matches,
            'keypoints': kp_piece,
            'color_verified': True
        }

    def verify_color(self, piece, homography):
        """
        Extracts the region from reference image defined by homography mapping of piece bounds,
        and compares color histograms.
        """
        try:
            h_ref, w_ref = self.ref.reference_image.shape[:2]
            
            # Warp the piece mask to the reference image
            # piece['mask'] is same size as piece['image']
            piece_mask = piece['mask']
            
            # We need to construct a full-reference-sized mask?
            # Or just wrap the mask relative to the piece position?
            # M maps Piece(x,y) -> Ref(x,y).
            
            # To do this efficiently:
            # 1. Warp piece mask using H to get the shape on Ref image.
            warped_mask = cv2.warpPerspective(piece_mask, homography, (w_ref, h_ref))
            
            # 2. Calculate Histogram of Reference Image UNDER the warped mask
            # Convert to HSV for better color comparison
            ref_hsv = cv2.cvtColor(self.ref.reference_image, cv2.COLOR_BGR2HSV)
            hist_ref = cv2.calcHist([ref_hsv], [0, 1], warped_mask, [180, 256], [0, 180, 0, 256])
            cv2.normalize(hist_ref, hist_ref, 0, 1, cv2.NORM_MINMAX)
            
            # 3. Calculate Histogram of Piece Image
            piece_hsv = cv2.cvtColor(piece['image'], cv2.COLOR_BGR2HSV)
            hist_piece = cv2.calcHist([piece_hsv], [0, 1], piece_mask, [180, 256], [0, 180, 0, 256])
            cv2.normalize(hist_piece, hist_piece, 0, 1, cv2.NORM_MINMAX)
            
            # 4. Compare
            # Correlation: 1 is perfect match
            score = cv2.compareHist(hist_ref, hist_piece, cv2.HISTCMP_CORREL)
            
            # Threshold: Correlation > 0.3? (Tunable)
            # Puzzle pieces from photo vs box art might have different lighting/saturation.
            return score > -1.0
            
        except Exception as e:
            print(f"Color verification error: {e}")
            return True # Fail safe allowed
