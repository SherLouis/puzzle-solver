import cv2
import numpy as np
import os

def load_image(path):
    """Loads an image from the specified path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at {path}")
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to load image at {path}")
    return image

def preprocess_for_segmentation(image):
    """
    Preprocesses the image for segmentation.
    Returns a grayscale image/mask that highlights the pieces.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred
