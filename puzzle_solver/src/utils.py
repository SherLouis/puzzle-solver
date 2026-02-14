import cv2
import matplotlib.pyplot as plt
import numpy as np

def show_image(image, title="Image", cmap=None):
    """Displays an image using matplotlib."""
    plt.figure(figsize=(10, 10))
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert BGR to RGB for matplotlib
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    """Resizes an image maintaining aspect ratio."""
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized
