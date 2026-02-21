import cv2
import numpy as np

def explore_channels():
    image_path = 'data/pieces.jpg'
    original_img = cv2.imread(image_path)

    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(original_img, cv2.COLOR_BGR2LAB)
    
    channels = {
        'Gray': gray,
        'H': hsv[:,:,0],
        'S': hsv[:,:,1],
        'V': hsv[:,:,2],
        'L': lab[:,:,0],
        'A': lab[:,:,1],
        'B': lab[:,:,2]
    }
    
    for name, ch in channels.items():
        # blur
        blurred = cv2.GaussianBlur(ch, (9, 9), 0)
        
        # otsu
        ret, otsu1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret, otsu2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        for idx, otsu in enumerate([otsu1, otsu2]):
            # morphology
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=2)
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)
            
            contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_contours = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 1000 < area < 100000:
                    valid_contours.append(area)
            
            if len(valid_contours) > 20:
                print(f"Channel {name} Otsu {idx+1}: Found {len(valid_contours)} pieces")

if __name__ == '__main__':
    explore_channels()
