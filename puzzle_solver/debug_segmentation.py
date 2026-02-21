import cv2
import numpy as np
import os
from src.image_processing import load_image

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def visualize_segmentation():
    image_path = 'puzzle_solver/data/pieces.jpg'
    output_dir = 'puzzle_solver/segmentation_steps'
    ensure_dir(output_dir)
    
    # Load Image
    image = load_image(image_path)
    if image is None:
        print("Failed to load image")
        return
    cv2.imwrite(f"{output_dir}/01_original.jpg", image)
    
    # 1. Grayscale & Blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{output_dir}/02_gray.jpg", gray)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite(f"{output_dir}/03_blurred.jpg", blurred)
    
    # 2. Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 51, 2)
    cv2.imwrite(f"{output_dir}/04_adaptive_thresh.jpg", thresh)
    
    # 3. Morphological Operations
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imwrite(f"{output_dir}/05_morph_open.jpg", opening)
    
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    cv2.imwrite(f"{output_dir}/06_morph_close.jpg", closing)
    
    # 4. Contours
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contour_img = image.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(f"{output_dir}/07_all_contours.jpg", contour_img)
    
    # 5. Filtering
    valid_contours = []
    min_area = 500
    max_area = 100000
    
    filtered_img = image.copy()
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            valid_contours.append(contour)
            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(filtered_img, (x,y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(filtered_img, str(i), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    cv2.imwrite(f"{output_dir}/08_filtered_contours_{len(valid_contours)}_pieces.jpg", filtered_img)
    
    print(f"Saved {len(valid_contours)} valid pieces.")
    
    # 6. Wood Color Removal Visualization (Alternative Strategy)
    # Just to compare
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_wood = np.array([5, 40, 50])    
    upper_wood = np.array([25, 255, 255])
    bg_mask = cv2.inRange(hsv, lower_wood, upper_wood)
    fg_mask = cv2.bitwise_not(bg_mask)
    cv2.imwrite(f"{output_dir}/09_hsv_wood_removal_mask.jpg", fg_mask)

if __name__ == "__main__":
    visualize_segmentation()
