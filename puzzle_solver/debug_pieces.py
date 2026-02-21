import cv2
import os
from src.segmentation import PieceDetector
from src.image_processing import load_image

def debug_pieces():
    pieces_path = 'puzzle_solver/data/pieces.jpg'
    pieces_img = load_image(pieces_path)
    
    if pieces_img is None:
        print("Failed to load image")
        return

    print("Detecting pieces...")
    detector = PieceDetector()
    pieces = detector.detect_pieces(pieces_img)
    
    print(f"Found {len(pieces)} pieces.")
    
    # Save each piece
    output_dir = 'debug_pieces_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for i, piece in enumerate(pieces):
        filename = f"{output_dir}/piece_{i}.jpg"
        cv2.imwrite(filename, piece['image'])
        
        # Also save mask
        mask_filename = f"{output_dir}/piece_{i}_mask.jpg"
        cv2.imwrite(mask_filename, piece['mask'])
        
    print(f"Saved {len(pieces)} pieces to {output_dir}")

if __name__ == "__main__":
    debug_pieces()
