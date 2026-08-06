import cv2
import numpy as np
import os
import time

QA_LOG_DIR = os.path.join(os.path.dirname(__file__), "qa_logs")

def is_ir_frame(frame):
    """
    Checks if frame comes from an Infrared/Grayscale night vision camera.
    Returns True if color channels (R, G, B) are identical (mean channel diff < 3).
    """
    if len(frame.shape) == 2:
        return True
    b, g, r = cv2.split(frame)
    diff_rg = np.mean(np.abs(r.astype(np.float32) - g.astype(np.float32)))
    diff_gb = np.mean(np.abs(g.astype(np.float32) - b.astype(np.float32)))
    return (diff_rg + diff_gb) < 3.0

def detect_low_light(frame, threshold=75.0):
    """
    Calculates mean luminance. Returns True if brightness < threshold.
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    mean_brightness = float(np.mean(gray))
    return mean_brightness < threshold, mean_brightness

def apply_gamma_correction(image, gamma=1.6):
    """Enhances dark shadows in low-light frames"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def apply_clahe(frame, clip_limit=3.0, tile_grid_size=(8, 8)):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    For color frames, applies to LAB 'L'-channel to prevent color distortion.
    For IR/grayscale frames, applies directly.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if len(frame.shape) == 3:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    else:
        return clahe.apply(frame)

def apply_deblur(frame, blur_threshold=110.0):
    """
    Measures Laplacian variance. If motion blur detected (var < blur_threshold),
    applies unsharp masking to sharpen plate character edges.
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
        
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance < blur_threshold:
        # Unsharp Masking Kernel
        gaussian = cv2.GaussianBlur(frame, (0, 0), 2.5)
        sharpened = cv2.addWeighted(frame, 1.5, gaussian, -0.5, 0)
        return sharpened, True, round(float(variance), 2)
    return frame, False, round(float(variance), 2)

def enhance_frame(frame, log_qa=False, save_name="frame"):
    """
    Unified Pre-Processing & Enhancement Pipeline:
    1. IR / Grayscale Camera Feed Handling.
    2. Low-Light Detection & Adaptive Gamma Enhancement.
    3. CLAHE Glare & Shadow Correction.
    4. Motion-Blur Detection & Edge Sharpening.
    5. QA Image Logging (if log_qa=True).
    """
    if frame is None or frame.size == 0:
        return frame

    original_frame = frame.copy()
    actions_taken = []

    # 1. IR / Grayscale Camera Feed Handling
    ir_mode = is_ir_frame(frame)
    if ir_mode:
        actions_taken.append("IR_CAMERA_MODE")

    # 2. Low-Light Detection & Adaptive Gamma Enhancement
    is_dark, brightness = detect_low_light(frame, threshold=75.0)
    if is_dark:
        frame = apply_gamma_correction(frame, gamma=1.7)
        actions_taken.append(f"GAMMA_BOOST(brightness={brightness:.1f})")

    # 3. CLAHE Glare & Shadow Correction
    frame = apply_clahe(frame, clip_limit=3.0)
    actions_taken.append("CLAHE_GLARE_CORRECTION")

    # 4. Motion-Blur Detection & Deblurring
    frame, was_deblurred, blur_var = apply_deblur(frame, blur_threshold=110.0)
    if was_deblurred:
        actions_taken.append(f"DEBLUR_SHARPENED(var={blur_var})")

    # 5. QA Review Logging
    if log_qa:
        os.makedirs(QA_LOG_DIR, exist_ok=True)
        h1, w1 = original_frame.shape[:2]
        if len(original_frame.shape) == 2:
            orig_bgr = cv2.cvtColor(original_frame, cv2.COLOR_GRAY2BGR)
            enh_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if len(frame.shape) == 2 else frame
        else:
            orig_bgr = original_frame
            enh_bgr = frame if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        cv2.putText(orig_bgr, "ORIGINAL INPUT", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(enh_bgr, f"ENHANCED: {', '.join(actions_taken)}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        comparison = np.hstack((orig_bgr, enh_bgr))
        timestamp = int(time.time())
        qa_path = os.path.join(QA_LOG_DIR, f"qa_{save_name}_{timestamp}.jpg")
        cv2.imwrite(qa_path, comparison)
        print(f"[FrameEnhancer] Logged QA side-by-side review image to: {qa_path}")

    return frame
