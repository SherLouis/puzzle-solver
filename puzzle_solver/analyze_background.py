import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def analyze_background(image_path, k=5):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize for speed
    small_img = cv2.resize(image_rgb, (200, 200))
    pixels = small_img.reshape(-1, 3)
    
    # Find dominant colors using K-Means
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # Count frequency of each cluster
    counts = np.bincount(labels)
    total = sum(counts)
    
    # Sort by frequency (most common first -> likely background)
    sorted_indices = np.argsort(counts)[::-1]
    sorted_colors = colors[sorted_indices]
    sorted_counts = counts[sorted_indices]
    
    print("Dominant Colors (RGB):")
    fig, axes = plt.subplots(1, k + 1, figsize=(15, 3))
    
    # Visualization of dominant colors
    bar = np.zeros((50, 300, 3), dtype=np.uint8)
    start_x = 0
    
    for i in range(k):
        color = sorted_colors[i].astype(int)
        percent = sorted_counts[i] / total
        print(f"Color {i}: {color} - {percent*100:.1f}%")
        
        # Display color patch
        patch = np.ones((100, 100, 3), dtype=np.uint8) * color
        axes[i].imshow(patch)
        axes[i].set_title(f"{percent*100:.0f}%")
        axes[i].axis('off')
        
    print("\n--- Background Removal Test ---")
    # Assumption: The most frequent color is the background (wood table)
    bg_color = sorted_colors[0].astype(np.uint8)
    
    # Convert to HSV to define a range around this color
    # bg_patch must be (1, 1, 3) image
    bg_patch = np.uint8([[bg_color]])
    bg_hsv = cv2.cvtColor(bg_patch, cv2.COLOR_RGB2HSV)[0][0]
    
    print(f"Background HSV: {bg_hsv}")
    
    # Define range (e.g., +/- 20 hue, +/- 50 sat/val)
    # Wood is usually Orange/Brown (Hue 10-20)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Sensitivity
    hue_thresh = 20
    sat_thresh = 50
    val_thresh = 50
    
    lower = np.array([max(0, bg_hsv[0] - hue_thresh), max(0, bg_hsv[1] - sat_thresh), max(0, bg_hsv[2] - val_thresh)])
    upper = np.array([min(180, bg_hsv[0] + hue_thresh), min(255, bg_hsv[1] + sat_thresh), min(255, bg_hsv[2] + val_thresh)])
    
    mask = cv2.inRange(img_hsv, lower, upper)
    
    # The mask contains the background (white). We want the inverse (pieces).
    fg_mask = cv2.bitwise_not(mask)
    
    # Clean up
    kernel = np.ones((5,5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    axes[k].imshow(fg_mask, cmap='gray')
    axes[k].set_title("Result Mask")
    axes[k].axis('off')
    
    plt.savefig("background_analysis.png")
    cv2.imwrite("debug_bg_mask.jpg", fg_mask)
    print("Saved 'background_analysis.png' and 'debug_bg_mask.jpg'")

if __name__ == "__main__":
    analyze_background('puzzle_solver/data/pieces.jpg')
