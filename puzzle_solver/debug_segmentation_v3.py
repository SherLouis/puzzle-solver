import cv2
import numpy as np
import os

def visualize_segmentation_v3():
    image_path = 'puzzle_solver/data/pieces.jpg'
    output_dir = 'puzzle_solver/segmentation_v3_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Sweep Saturation Lower Bound
    # Wood is likely high saturation. Pieces might be lower? 
    # Or Wood is specific Hue.
    
    # Let's try to capture "Everything that is NOT the table".
    # Table seems to be Hue=15-25, Sat=150-255, Val=100-255.
    
    configs = [
        ("Base", [5, 50, 50], [30, 255, 255]),
        ("HighSat", [5, 100, 50], [30, 255, 255]), # Wood is very saturated
        ("WideHue", [0, 50, 50], [40, 255, 255]),
        ("DarkWood", [5, 40, 20], [30, 255, 180]),
    ]
    
    for name, lower, upper in configs:
        lower = np.array(lower)
        upper = np.array(upper)
        
        mask = cv2.inRange(hsv, lower, upper)
        # Invert -> Pieces
        pieces = cv2.bitwise_not(mask)
        
        # Clean
        kernel = np.ones((5,5), np.uint8)
        pieces = cv2.morphologyEx(pieces, cv2.MORPH_OPEN, kernel, iterations=2)
        pieces = cv2.morphologyEx(pieces, cv2.MORPH_CLOSE, kernel, iterations=5) # Aggressive closing
        
        # Contours
        contours, _ = cv2.findContours(pieces, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        display = image.copy()
        count = 0
        mean_area = 0
        areas = []
        for c in contours:
            area = cv2.contourArea(c)
            if 1000 < area < 50000:
                count += 1
                areas.append(area)
                cv2.drawContours(display, [c], -1, (0, 255, 0), 2)
        
        if areas: mean_area = int(np.mean(areas))
        
        cv2.putText(display, f"{name}: {count} pieces, AvgArea: {mean_area}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(f"{output_dir}/test_{name}.jpg", display)
        print(f"Config {name}: Found {count} pieces.")

if __name__ == "__main__":
    visualize_segmentation_v3()
