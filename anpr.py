import cv2
import easyocr
import argparse
import csv
import os
import re
from collections import Counter

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
    for i in range(2):
        if chars[i] in num_to_char:
            chars[i] = num_to_char[chars[i]]
            
    # 2. RTO Code (Next 2 characters)
    for i in range(2, min(4, len(chars))):
        if chars[i] in char_to_num:
            chars[i] = char_to_num[chars[i]]

    # 3. Last 4 characters
    for i in range(len(chars)-4, len(chars)):
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

def process_video(video_path, output_path, csv_path, min_confidence=0.15, skip_frames=2):
    detector = load_plate_detector()
    reader = load_ocr_reader()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_number", "timestamp_sec", "plate_text", "confidence"])

    frame_count = 0
    detections_log = []

    print(f"Processing video: {video_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count % skip_frames == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            plates = detector.detectMultiScale(gray_frame, scaleFactor=1.05, minNeighbors=2, minSize=(30, 10))

            for (x, y, w, h) in plates:
                aspect_ratio = w / float(h)
                if aspect_ratio < 1.5 or aspect_ratio > 6.0:
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
                    cleaned = format_indian_plate(text)
                    if cleaned and conf >= min_confidence:
                        timestamp = frame_count / fps
                        csv_writer.writerow([frame_count, round(timestamp, 2), cleaned, round(conf, 2)])
                        detections_log.append(cleaned)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{cleaned} ({conf:.2f})"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out.write(frame)
        if frame_count % 50 == 0:
            print(f"  ...processed {frame_count} frames")

    cap.release()
    out.release()
    csv_file.close()

    print(f"\nDone. Processed {frame_count} frames.")
    print(f"Annotated video saved to: {output_path}")
    print(f"Detection log saved to: {csv_path}")

    if detections_log:
        most_common, count = Counter(detections_log).most_common(1)[0]
        print(f"\nMost frequently read plate: {most_common}  (seen {count} times)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect ONLY Indian number plates from a video.")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", default="output_annotated.mp4", help="Path to save annotated output video")
    parser.add_argument("--csv", default="detections.csv", help="Path to save detection log CSV")
    parser.add_argument("--min-confidence", type=float, default=0.15, help="Minimum OCR confidence to accept a reading")
    parser.add_argument("--skip-frames", type=int, default=2, help="Run detection every Nth frame")
    args = parser.parse_args()

    process_video(args.video, args.output, args.csv, args.min_confidence, args.skip_frames)
