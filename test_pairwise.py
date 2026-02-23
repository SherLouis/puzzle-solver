import cv2
import numpy as np
import itertools
from src.image_processing import load_image
from src.segmentation import PieceDetector

def extract_sides(contour, center):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int64(box)
    
    # order box points to be predictable (e.g. top-left clockwise)
    # Actually just iterate around the contour and split it into 4 segments based on the corners of the box.
    # To find corners on the contour, find the 4 points on the contour closest to the 4 box points.
    contour_pts = contour[:, 0, :]
    corners = []
    for bp in box:
        dists = np.linalg.norm(contour_pts - bp, axis=1)
        corners.append(np.argmin(dists))
        
    # sort the corners by their index in the contour
    corners = sorted(corners)
    
    sides = []
    for i in range(4):
        start_idx = corners[i]
        end_idx = corners[(i + 1) % 4]
        
        if start_idx < end_idx:
            side_pts = contour_pts[start_idx:end_idx+1]
        else:
            side_pts = np.vstack((contour_pts[start_idx:], contour_pts[:end_idx+1]))
            
        # Is it a straight edge, knob, or hole?
        # A straight line from start to end
        p1 = contour_pts[start_idx]
        p2 = contour_pts[end_idx]
        length = np.linalg.norm(p1 - p2)
        if length == 0: continue
            
        edge_dir = (p2 - p1) / length
        normal = np.array([-edge_dir[1], edge_dir[0]])
        
        # calculate signed distance of all points to the line
        # line eq: (p-p1) dot normal = 0
        v = side_pts - p1
        dists = np.dot(v, normal)
        
        max_dist = np.max(dists)
        min_dist = np.min(dists)
        
        # Classify based on max deviation
        if max_dist < 10 and min_dist > -10:
            type_ = 'flat'
        else:
            # check the direction of the center to orient the normal
            v_center = center - p1
            center_dist = np.dot(v_center, normal)
            
            # if max deviation is towards the center, it's a hole. If away, it's a knob.
            # let's just save the normalized profile
            type_ = 'curved'
            
        sides.append({
            'pts': side_pts,
            'p1': p1,
            'p2': p2,
            'length': length,
            'max_dist': max_dist,
            'min_dist': min_dist,
            'dists': dists,
            'type': type_
        })
    return sides

img = load_image('puzzle_solver/data/pieces.jpg')
d = PieceDetector()
pieces = d.detect_pieces(img)

# Filter edge/corner pieces
border_pieces = [p for p in pieces if p.get('type') in ['edge', 'corner']]
print(f"Found {len(border_pieces)} border pieces.")

for i, p in enumerate(border_pieces[:3]):
    sides = extract_sides(p['contour'], p['center'])
    print(f"Piece {i}: {[s['type'] for s in sides]}")

