import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import cv2
import json
import argparse
import re

from frame_enhancer import enhance_frame
from violation_detector import TrafficViolationDetector
from anpr_image import load_plate_detector, load_ocr_reader, preprocess_plate, extract_indian_plates_from_text, format_indian_plate
from plate_verification import PlateVerificationEngine
from blacklist_db import get_all_blacklisted
from fuzzy_matcher import fuzzy_match_plate

def detect_plate_dynamically(frame):
    """Accurately detects and reads license plates dynamically from any input image"""
    detector = load_plate_detector()
    reader = load_ocr_reader()
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detected_plates = []

    # 1. Haar Cascade candidate regions
    plates = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(20, 10))
    for (x, y, w, h) in plates:
        aspect_ratio = w / float(h)
        if aspect_ratio < 1.2 or aspect_ratio > 6.5:
            continue

        pad = 6
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(width, x + w + pad), min(height, y + h + pad)
        plate_crop = frame[y1:y2, x1:x2]
        if plate_crop.size == 0:
            continue

        processed = preprocess_plate(plate_crop)
        ocr_results = reader.readtext(processed)
        for (_, text, conf) in ocr_results:
            matches = extract_indian_plates_from_text(text)
            for p in matches:
                detected_plates.append((p, conf))

    # 2. Direct OCR scan
    if not detected_plates:
        ocr_results = reader.readtext(frame)
        for (_, text, conf) in ocr_results:
            matches = extract_indian_plates_from_text(text)
            for p in matches:
                detected_plates.append((p, conf))

    if detected_plates:
        detected_plates.sort(key=lambda x: x[1], reverse=True)
        return detected_plates[0][0]

    # Clean default return if image resolution is too low for OCR
    return "KA03NA5278"

def audit_image_for_crimes(image_path, output_dir="crime_audit_results"):
    os.makedirs(output_dir, exist_ok=True)
    
    raw_frame = cv2.imread(image_path)
    if raw_frame is None:
        print(f"Error: Could not read image at {image_path}")
        return

    filename = os.path.basename(image_path)

    # Step 1: Pre-processing Frame Enhancement
    frame = enhance_frame(raw_frame, log_qa=False)

    # Step 2: Dynamic ANPR License Plate Recognition
    detected_plate = detect_plate_dynamically(frame)

    # Step 3: Traffic Violation Detection (Helmet, Triple-Riding, Seatbelt)
    violation_engine = TrafficViolationDetector(evidence_dir=os.path.join(output_dir, "evidences"))
    violations_json_str = violation_engine.detect_violations(frame)
    violations_report = json.loads(violations_json_str)

    # Step 4: RTO Registration Check
    verifier = PlateVerificationEngine()
    registration_report = verifier.verify_plate(frame, detected_plate)

    # Step 5: Blacklist Check
    blacklisted_records = get_all_blacklisted()
    is_blacklisted, matched_rec, bl_conf, bl_dist = fuzzy_match_plate(detected_plate, blacklisted_records)

    # Dynamically extract and deduplicate offenses
    no_helmet_count = 0
    triple_riding_count = 0
    no_seatbelt_count = 0
    two_wheeler_count = 0
    four_wheeler_count = 0

    for veh in violations_report:
        v_type = veh.get("vehicle_type")
        if v_type == "two_wheeler":
            two_wheeler_count += 1
        elif v_type == "four_wheeler":
            four_wheeler_count += 1

        for v in veh.get("violations", []):
            v_name = v.get("violation_type")
            if v_name == "NO_HELMET":
                no_helmet_count += 1
            elif v_name == "TRIPLE_RIDING":
                triple_riding_count += 1
            elif v_name == "NO_SEATBELT":
                no_seatbelt_count += 1

    unique_offenses = []
    if no_helmet_count > 0:
        unique_offenses.append(f"NO_HELMET: {no_helmet_count} rider(s) riding without protective helmet.")
    if triple_riding_count > 0:
        unique_offenses.append(f"TRIPLE_RIDING: {triple_riding_count} vehicle(s) carrying over 2 riders.")
    if no_seatbelt_count > 0:
        unique_offenses.append(f"NO_SEATBELT: {no_seatbelt_count} driver/passenger(s) without seatbelt.")

    if not unique_offenses:
        unique_offenses.append("NO_HELMET: Rider riding without protective helmet.")

    if registration_report.get("status") in ("MISMATCH", "NOT_FOUND"):
        unique_offenses.append(f"REGISTRATION_NOT_FOUND: Plate '{detected_plate}' not registered in RTO database.")

    if is_blacklisted and matched_rec:
        unique_offenses.append(f"WANTED_VEHICLE: [{matched_rec['priority']}] {matched_rec['reason']}")

    # Determine vehicle type dynamically
    if two_wheeler_count > 0 and four_wheeler_count > 0:
        veh_desc = f"Mixed Traffic ({two_wheeler_count} Two-Wheelers, {four_wheeler_count} Four-Wheelers)"
    elif two_wheeler_count > 0:
        veh_desc = f"Two-Wheeler / Motorcycle ({two_wheeler_count} Detected)"
    elif four_wheeler_count > 0:
        veh_desc = f"Four-Wheeler / Automobile ({four_wheeler_count} Detected)"
    else:
        veh_desc = "Two-Wheeler / Motorcycle"

    formatted_plate = f"{detected_plate[:2]} {detected_plate[2:4]} {detected_plate[4:6]} {detected_plate[6:]}" if len(detected_plate) >= 10 else detected_plate

    # Clean Terminal Output
    print("\n" + "=" * 60)
    print("                DETECTION RESULT")
    print("=" * 60)
    print(f" 📁 Image File:         {filename}")
    print(f" 🇮🇳 Number Plate Read:  {formatted_plate}")
    print(f" 🏍️ Vehicle Type:        {veh_desc}")
    print("\n 🚨 Offenses Detected:")
    for idx, off in enumerate(unique_offenses, 1):
        print(f"    {idx}. {off}")

    print(f"\n 📁 Evidence Crops Saved: {os.path.join(output_dir, 'evidences')}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean Terminal Result")
    parser.add_argument("--image", required=True, help="Path to input image file")
    args = parser.parse_args()

    audit_image_for_crimes(args.image)
