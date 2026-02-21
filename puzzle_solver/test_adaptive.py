import cv2
import numpy as np

def test_adaptive_large_block():
    image_path = 'data/pieces.jpg'
    original_img = cv2.imread(image_path)
    if original_img is None:
        return
        
    lab = cv2.cvtColor(original_img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:,:,0]
    
    # Bilateral Blur
    blurred = cv2.bilateralFilter(l_channel, 9, 75, 75)
    
    for bs in [101, 301, 501, 701, 999]:
        for C in [-10, -5, 0, 5, 10]:
            try:
                thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY_INV, bs, C)
                                               
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
                closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=3)
                
                contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                valid_contours = []
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if 1500 < area < 100000:
                        valid_contours.append(area)
                
                if len(valid_contours) >= 48:
                    print(f"Adaptive L-Channel (bs={bs}, C={C}): Found {len(valid_contours)} pieces")
            except Exception as e:
                pass

if __name__ == '__main__':
    test_adaptive_large_block()
