import cv2
import easyocr
import argparse
import os
import re

CASCADE_PATH = os.path.join(os.path.dirname(__file__), "haarcascade_russian_plate_number.xml")

INDIAN_STATE_CODES = {
    'AN', 'AP', 'AR', 'AS', 'BR', 'CH', 'CG', 'DD', 'DN', 'DL', 'GA', 'GJ', 
    'HR', 'HP', 'JK', 'JH', 'KA', 'KL', 'LA', 'LD', 'MP', 'MH', 'MN', 'ML', 
    'MZ', 'NL', 'OD', 'PY', 'PB', 'RJ', 'SK', 'TN', 'TS', 'TR', 'UP', 'UK', 'WB'
}

def load_plate_detector():
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        raise IOError(f"Could not load cascade classifier from {CASCADE_PATH}")
    return detector

def load_ocr_reader():
    return easyocr.Reader(['en'], gpu=False, verbose=False)

def preprocess_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    gray = cv2.equalizeHist(gray)
    h, w = gray.shape
    if w < 300:
        scale = 300 / w
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    return gray

def format_indian_plate(text):
    clean = re.sub(r'[^A-Za-z0-9]', '', text).upper()
    if len(clean) < 8 or len(clean) > 11:
        return None

    char_to_num = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5', 'B': '8', 'Z': '2', 'Q': '0'}
    num_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S', '8': 'B', '2': 'Z'}

    chars = list(clean)
    
    # 1. State code (First 2 characters)
    for i in range(min(2, len(chars))):
        if chars[i] in num_to_char:
            chars[i] = num_to_char[chars[i]]
            
    # 2. RTO Code (Next 2 characters)
    for i in range(2, min(4, len(chars))):
        if chars[i] in char_to_num:
            chars[i] = char_to_num[chars[i]]

    # 3. Last 4 characters
    for i in range(max(0, len(chars)-4), len(chars)):
        if chars[i] in char_to_num:
            chars[i] = char_to_num[chars[i]]

    corrected = "".join(chars)
    
    # Check valid Indian state code prefix
    if corrected[:2] not in INDIAN_STATE_CODES:
        return None

    # Strict Indian number plate regex
    pattern = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$')
    if pattern.match(corrected):
        return corrected

    return None

def extract_indian_plates_from_text(raw_text):
    # Extracts any embedded Indian plate sequences from longer text
    tokens = re.findall(r'[A-Z0-9]{8,11}', raw_text.upper())
    found = []
    for token in tokens:
        formatted = format_indian_plate(token)
        if formatted:
            found.append(formatted)
    return found

from frame_enhancer import enhance_frame

def process_image(image_path, output_path=None, min_confidence=0.05, log_qa=False):
    detector = load_plate_detector()
    reader = load_ocr_reader()

    raw_frame = cv2.imread(image_path)
    if raw_frame is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Pass frame through pre-processing & enhancement engine
    frame = enhance_frame(raw_frame, log_qa=log_qa, save_name=os.path.basename(image_path))

    height, width = frame.shape[:2]
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    detected_plates = set()

    print(f"\nProcessing Image: {image_path}")

    # Pass 1: Haar Cascade candidate regions
    plates = detector.detectMultiScale(gray_frame, scaleFactor=1.05, minNeighbors=2, minSize=(20, 10))
    print(f"Haar Candidate regions: {len(plates)}")

    for (x, y, w, h) in plates:
        aspect_ratio = w / float(h)
        if aspect_ratio < 1.3 or aspect_ratio > 6.5:
            continue

        pad = 8
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(width, x + w + pad), min(height, y + h + pad)
        plate_crop = frame[y1:y2, x1:x2]
        if plate_crop.size == 0:
            continue

        processed = preprocess_plate(plate_crop)
        ocr_results = reader.readtext(processed)

        for (_, text, conf) in ocr_results:
            matches = extract_indian_plates_from_text(text)
            for plate in matches:
                if plate not in detected_plates and conf >= min_confidence:
                    detected_plates.add(plate)
                    print(f" [Haar + OCR] Found Plate: '{plate}' (Raw: '{text}', Conf: {conf:.2f})")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, f"{plate} ({conf:.2f})", (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Pass 2: Full-Image Direct OCR scan to catch plates missed by Haar Cascade
    print(" Running full-image OCR scan for missed plates...")
    ocr_results = reader.readtext(frame)
    for (bbox, text, conf) in ocr_results:
        matches = extract_indian_plates_from_text(text)
        for plate in matches:
            if plate not in detected_plates and conf >= min_confidence:
                detected_plates.add(plate)
                print(f" [Direct OCR Scan] Found Plate: '{plate}' (Raw: '{text}', Conf: {conf:.2f})")
                pt1 = (int(bbox[0][0]), int(bbox[0][1]))
                pt2 = (int(bbox[2][0]), int(bbox[2][1]))
                cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 3)
                cv2.putText(frame, f"{plate} ({conf:.2f})", (pt1[0], max(25, pt1[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    print(f"\nTotal Unique License Plates Detected: {len(detected_plates)}")
    for p in detected_plates:
        print(f"  - {p}")

    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_detected{ext}"

    cv2.imwrite(output_path, frame)
    print(f"\nSaved annotated image to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect ALL Indian number plates from an image.")
    parser.add_argument("--image", required=True, help="Path to input image file")
    parser.add_argument("--output", help="Path to save annotated output image")
    parser.add_argument("--min-confidence", type=float, default=0.05, help="Minimum OCR confidence")
    args = parser.parse_args()

    process_image(args.image, args.output, args.min_confidence)
