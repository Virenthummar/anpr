import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import re
import os

from crnn_model import CRNN, ctc_decode, ALPHABET
from plate_deskew import deskew_plate

INDIAN_STATE_CODES = {
    'AN', 'AP', 'AR', 'AS', 'BR', 'CH', 'CG', 'DD', 'DN', 'DL', 'GA', 'GJ', 
    'HR', 'HP', 'JK', 'JH', 'KA', 'KL', 'LA', 'LD', 'MP', 'MH', 'MN', 'ML', 
    'MZ', 'NL', 'OD', 'PY', 'PB', 'RJ', 'SK', 'TN', 'TS', 'TR', 'UP', 'UK', 'WB'
}

class CustomIndianOCREngine:
    """Custom Indian OCR Engine (CRNN + Layout Handler + Deskewing + Indian Post-Processing)"""
    def __init__(self, crnn_checkpoint_path="crnn_indian_plate.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Load CRNN Model
        self.crnn = CRNN(nc=1, nclass=len(ALPHABET)).to(self.device)
        if crnn_checkpoint_path and os.path.exists(crnn_checkpoint_path):
            try:
                self.crnn.load_state_dict(torch.load(crnn_checkpoint_path, map_location=self.device))
                print(f"[CustomOCR] Loaded CRNN weights from {crnn_checkpoint_path}")
            except Exception as e:
                print(f"[CustomOCR] Warning: Could not load {crnn_checkpoint_path}: {e}")
        self.crnn.eval()

        # EasyOCR Fallback Engine
        try:
            import easyocr
            self.fallback_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        except Exception:
            self.fallback_reader = None

    def recognize_single_line(self, crop):
        """Runs CRNN model on single image line crop"""
        if crop is None or crop.size == 0:
            return "", 0.0

        tensor = self.transform(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.crnn(tensor)
            pred_text, conf = ctc_decode(logits)

        # Fallback to EasyOCR if CRNN prediction is short or low confidence
        if len(pred_text) < 4 and self.fallback_reader:
            results = self.fallback_reader.readtext(crop)
            if results:
                _, fall_text, fall_conf = results[0]
                if fall_conf > conf:
                    return fall_text, round(float(fall_conf), 2)

        return pred_text, conf

    def detect_layout_and_recognize(self, plate_crop):
        """
        Handles Single-Line and Double-Line Stacked Layouts.
        If height/width ratio indicates a 2-line stacked plate (aspect ratio < 2.5),
        splits crop into top and bottom halves and concatenates recognized text.
        """
        h, w = plate_crop.shape[:2]
        aspect_ratio = w / float(h)

        # Double-Line Stacked Layout Check (e.g. 2-wheelers / commercial vehicles)
        if aspect_ratio < 2.5 and h >= 30:
            layout_type = "double_line"
            mid_h = int(h * 0.52)
            top_crop = plate_crop[0:mid_h, 0:w]
            bottom_crop = plate_crop[mid_h:h, 0:w]

            top_text, top_conf = self.recognize_single_line(top_crop)
            bottom_text, bottom_conf = self.recognize_single_line(bottom_crop)

            raw_combined = f"{top_text}{bottom_text}"
            avg_conf = round((top_conf + bottom_conf) / 2.0, 2)
            return raw_combined, avg_conf, layout_type

        # Single-Line Standard Layout
        layout_type = "single_line"
        raw_text, conf = self.recognize_single_line(plate_crop)
        return raw_text, conf, layout_type

    def post_process_indian_format(self, text):
        """
        Applies Indian State Code rules and Regex auto-correction:
        Auto-corrects 0/O, 1/I, 8/B based on character positions.
        """
        clean = re.sub(r'[^A-Za-z0-9]', '', text).upper()
        if len(clean) < 8 or len(clean) > 11:
            return clean, False

        char_to_num = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5', 'B': '8', 'Z': '2', 'Q': '0'}
        num_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S', '8': 'B', '2': 'Z'}

        chars = list(clean)

        # First 2 chars: State Code (Letters)
        for i in range(min(2, len(chars))):
            if chars[i] in num_to_char:
                chars[i] = num_to_char[chars[i]]

        # Next 2 chars: RTO Code (Digits)
        for i in range(2, min(4, len(chars))):
            if chars[i] in char_to_num:
                chars[i] = char_to_num[chars[i]]

        # Last 4 chars: Unique ID (Digits)
        for i in range(max(0, len(chars)-4), len(chars)):
            if chars[i] in char_to_num:
                chars[i] = char_to_num[chars[i]]

        corrected = "".join(chars)

        # Validate State Code
        if corrected[:2] not in INDIAN_STATE_CODES:
            return corrected, False

        # Validate Indian Regex Format: [A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}
        pattern = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$')
        is_valid = bool(pattern.match(corrected))

        return corrected, is_valid

    def process_plate_crop(self, plate_crop):
        """Full Pipeline: Deskewing -> Layout Handling -> CRNN -> Post-Processing"""
        # Step 1: Deskewing / Perspective Correction
        deskewed_crop, was_deskewed = deskew_plate(plate_crop)

        # Step 2 & 3: Layout Detection & Character Recognition
        raw_text, confidence, layout_type = self.detect_layout_and_recognize(deskewed_crop)

        # Step 4: Indian Format Post-Processing & Validation
        final_plate, is_valid_format = self.post_process_indian_format(raw_text)

        return {
            "plate_string": final_plate,
            "raw_ocr_string": raw_text,
            "confidence": confidence,
            "is_valid_format": is_valid_format,
            "layout_type": layout_type,
            "deskewed": was_deskewed
        }
