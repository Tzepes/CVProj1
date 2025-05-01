import cv2
import numpy as np

def is_tile_present(cell_img, threshold=40, min_foreground_ratio=0.05):
    """
    Determine if a tile is present in the given cell.
    
    :param cell_img: Cropped BGR image of a grid cell.
    :param threshold: Brightness threshold to isolate tile vs background.
    :param min_foreground_ratio: Minimum ratio of pixels that must be foreground.
    :return: True if tile is likely present, else False.
    """
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    
    # Apply binary threshold (tiles should be darker than white board)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # Compute ratio of foreground pixels (value 255)
    foreground_ratio = np.sum(binary == 255) / binary.size

    return foreground_ratio > min_foreground_ratio