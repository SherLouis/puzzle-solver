import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.image_processing import load_image, preprocess_for_segmentation
from src.segmentation import PieceDetector
from src.utils import show_image

# Load the image
image_path = 'puzzle_solver/data/pieces.jpg'
image = load_image(image_path)

# Initialize detector
detector = PieceDetector(min_area=100, max_area=500000) # Relaxed thresholds for debugging

print(f"Image shape: {image.shape}")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

methods = [
    ("Adaptive_51", lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 2)),
    ("Adaptive_101", lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 101, 2)),
    ("Adaptive_201", lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 201, 2)),
]

for name, func in methods:
    print(f"\n--- Method: {name} ---")
    thresh = func(blurred)
    cv2.imwrite(f"debug_thresh_{name}.jpg", thresh)
    
    # Morphological operations
    # Try slightly larger kernel to close gaps inside pieces
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter for "Piece-sized" objects
    # Assuming piece is roughly 50x50 to 200x200? 
    # Image is 1024x771. 50 pieces. 
    # Approx area = (1024*771) / 50 = 15000 pixels if they filled the screen.
    # But they are scattered.
    # Real piece area might be 2000-8000 pixels.
    piece_contours = [c for c in contours if 1000 < cv2.contourArea(c) < 20000]
    
    print(f"Total contours: {len(contours)}")
    print(f"Potential Pieces (1k-20k): {len(piece_contours)}")
    
    if piece_contours:
        areas = [cv2.contourArea(c) for c in piece_contours]
        print(f"Piece Areas: min={min(areas):.1f}, max={max(areas):.1f}, avg={np.mean(areas):.1f}")
        
    # Visualize
    output_image = image.copy()
    for c in piece_contours:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(f'detected_pieces_{name}.jpg', output_image)


detected_pieces = [] # clear for now until we pick a method

print(f"Detected {len(detected_pieces)} pieces.")

# Visualize results
output_image = image.copy()
for i, piece in enumerate(detected_pieces):
    x, y, w, h = piece['bbox']
    cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # text might be too cluttered, let's skip for now or make it small
    # cv2.putText(output_image, str(i), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

cv2.imwrite('detected_pieces_viz.jpg', output_image)
print("Saved detected pieces visualization to detected_pieces_viz.jpg")

# Save a few individual pieces
if detected_pieces:
    fig, axes = plt.subplots(1, min(5, len(detected_pieces)), figsize=(15, 3))
    if len(detected_pieces) == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        if i < len(detected_pieces):
            # Convert BGRA to RGBA
            img_rgba = cv2.cvtColor(detected_pieces[i]['image_alpha'], cv2.COLOR_BGRA2RGBA)
            ax.imshow(img_rgba)
            ax.axis('off')
    plt.savefig('extracted_pieces_sample.png')
    print("Saved extracted pieces sample to extracted_pieces_sample.png")
