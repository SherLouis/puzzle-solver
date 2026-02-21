import cv2
import numpy as np
import os

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def visualize_segmentation_v2():
    image_path = 'puzzle_solver/data/pieces.jpg'
    output_dir = 'puzzle_solver/segmentation_v2_results'
    ensure_dir(output_dir)
    
    # Load Image
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image")
        return
    
    # Save original
    cv2.imwrite(f"{output_dir}/00_original.jpg", image)
    
    # 1. HSV Conversion
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 2. Wood Detection (Background)
    # Wood is typically Orange/Brown.
    # Hue: 10-30 (Orange-ish)
    # Saturation: Moderate to High
    # Value: Moderate to High
    
    # Adjusting ranges based on "wood" assumption
    # Lower bound: Hue 5 (Red-Orange), Sat 50 (Not grey), Val 50 (Not dark)
    # Upper bound: Hue 35 (Yellow-Orange), Sat 255, Val 255
    lower_wood = np.array([0, 60, 60])     # More aggressive saturation filter
    upper_wood = np.array([40, 255, 255])
    
    wood_mask = cv2.inRange(hsv, lower_wood, upper_wood)
    cv2.imwrite(f"{output_dir}/01_wood_mask_raw.jpg", wood_mask)
    
    # 3. Invert to get "Potential Pieces"
    # Pieces are "Not Wood"
    pieces_mask = cv2.bitwise_not(wood_mask)
    cv2.imwrite(f"{output_dir}/02_pieces_mask_raw.jpg", pieces_mask)
    
    # 4. Clean up the mask
    # A. Remove small noise (wood grain specks)
    kernel_small = np.ones((3,3), np.uint8)
    pieces_mask = cv2.morphologyEx(pieces_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
    cv2.imwrite(f"{output_dir}/03_pieces_mask_opened.jpg", pieces_mask)
    
    # B. Fill holes inside pieces (force solidity)
    # Use a LARGE kernel because pieces are solid objects
    kernel_large = np.ones((7,7), np.uint8)
    pieces_mask_closed = cv2.morphologyEx(pieces_mask, cv2.MORPH_CLOSE, kernel_large, iterations=4)
    cv2.imwrite(f"{output_dir}/04_pieces_mask_closed.jpg", pieces_mask_closed)
    
    # 5. Find Contours on the clean mask
    contours, _ = cv2.findContours(pieces_mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    min_area = 1000 # Ignore tiny specks
    max_area = 500000 # Ignore huge chunks (entire table??)
    
    debug_img = image.copy()
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            # Aspect Ratio Check (Puzzle pieces aren't long strips)
            x,y,w,h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h
            if 0.2 < aspect_ratio < 5.0:
                valid_contours.append(contour)
                cv2.rectangle(debug_img, (x,y), (x+w, y+h), (0, 255, 0), 2)
                cv2.drawContours(debug_img, [contour], -1, (0, 0, 255), 2)
                cv2.putText(debug_img, f"P{i}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite(f"{output_dir}/05_final_contours.jpg", debug_img)
    print(f"Found {len(valid_contours)} pieces using Improved HSV strategy.")

if __name__ == "__main__":
    visualize_segmentation_v2()
