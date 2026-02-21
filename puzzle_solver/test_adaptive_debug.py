import cv2
import numpy as np

def test_adaptive_again():
    image_path = 'data/pieces.jpg'
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    for bs in [51, 99, 151, 301, 501]:
        try:
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, bs, 10)
                                           
            # morphology
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            count = 0
            for c in contours:
                if cv2.contourArea(c) > 2000:
                    count += 1
            print(f"Adaptive (bs={bs}, C=10): {count} pieces.")
        except Exception as e:
            print(f"Error at bs={bs}: {e}")

if __name__ == '__main__':
    test_adaptive_again()
