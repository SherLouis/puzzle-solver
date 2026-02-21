import cv2
import numpy as np

def test_kmeans():
    image_path = 'data/pieces.jpg'
    img = cv2.imread(image_path)
    
    Z = img.reshape((-1,3))
    Z = np.float32(Z)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    
    # Check both clusters
    for cluster_id in range(K):
        mask = np.uint8(label.reshape((img.shape[0], img.shape[1])) == cluster_id) * 255
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)
        
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        areas = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 500:
                areas.append(int(area))
                
        print(f"K-means Cluster {cluster_id}: {len(areas)} pieces, Top areas: {sorted(areas, reverse=True)[:5]}")

if __name__ == '__main__':
    test_kmeans()
