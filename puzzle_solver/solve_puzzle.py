import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.image_processing import load_image
from src.segmentation import PieceDetector
from src.matching import ReferenceAnalyzer, PieceMatcher
from src.utils import resize_image

def draw_matches(ref_image, matches):
    """
    Draws the matched pieces' bounding boxes on the reference image.
    matches: List of tuples (piece_index, homography)
    """
    output = ref_image.copy()
    
    for piece_idx, H, piece_info in matches:
        # Get piece dimensions (from bbox w, h)
        # We need the original piece contour or bbox to project it.
        # Let's project a simple box representing the piece frame.
        h, w = piece_info['bbox'][3], piece_info['bbox'][2]
        
        # Define corners of the piece in its local coordinate system
        # (Assuming the piece image passed to matcher was the crop)
        # The keypoints were detected in the crop.
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        
        try:
            dst = cv2.perspectiveTransform(pts, H)
            cv2.polylines(output, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
            
            # Draw centroid or label
            M = cv2.moments(dst)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(output, str(piece_idx), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        except Exception as e:
            print(f"Error drawing match for piece {piece_idx}: {e}")
            
    return output

def main():
    # 1. Load Images
    print("Loading images...")
    box_image = load_image('puzzle_solver/data/box.jpg')
    pieces_image = load_image('puzzle_solver/data/pieces.jpg')
    
    # 2. Segment Pieces
    print("Segmenting pieces...")
    detector = PieceDetector(min_area=1000, max_area=100000) # Using default/tuned params
    pieces = detector.detect_pieces(pieces_image)
    print(f"Found {len(pieces)} pieces.")
    
    # 3. Analyze Reference
    print("Analyzing reference...")
    ref_analyzer = ReferenceAnalyzer(box_image)
    kp, des = ref_analyzer.compute_features()
    print(f"Reference features: {len(kp)}")
    
    # 4. Match Pieces
    print("Matching pieces...")
    matcher = PieceMatcher(ref_analyzer)
    
    successful_matches = []
    
    for i, piece in enumerate(pieces):
        success, H, info = matcher.match_piece(piece)
        if success:
            print(f"Piece {i}: MATCHED ({info['matches']} matches)")
            successful_matches.append((i, H, piece))
        else:
            print(f"Piece {i}: No match ({info.get('matches', 0)} matches)")

    # 5. Visualize
    print(f"Drawing {len(successful_matches)} matches...")
    result_image = draw_matches(box_image, successful_matches)
    
    cv2.imwrite('solution_result.jpg', result_image)
    print("Saved solution result to 'solution_result.jpg'.")

if __name__ == "__main__":
    main()
