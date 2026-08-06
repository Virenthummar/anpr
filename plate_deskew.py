import cv2
import numpy as np

def deskew_plate(plate_crop):
    """
    Corrects rotation and perspective skew on license plate crops using Homography transformation.
    """
    if plate_crop is None or plate_crop.size == 0:
        return plate_crop, False

    h, w = plate_crop.shape[:2]
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    
    # Edge detection & Binarization
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find largest rectangular contour
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return plate_crop, False

    # Get contour with max area
    max_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(max_contour) < (w * h * 0.15):  # Ignore tiny noise
        return plate_crop, False

    # Find minimum area rotated rectangle
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    # Sort corners: top-left, top-right, bottom-right, bottom-left
    pts = np.float32(box)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    rect_pts = np.array([tl, tr, br, bl], dtype="float32")

    # Target output dimensions
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    if max_width < 30 or max_height < 10:
        return plate_crop, False

    dst_pts = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    # Perspective Transform Matrix
    matrix = cv2.getPerspectiveTransform(rect_pts, dst_pts)
    warped = cv2.warpPerspective(plate_crop, matrix, (max_width, max_height))

    return warped, True
