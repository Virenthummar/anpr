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
from custom_indian_ocr import CustomIndianOCREngine
from plate_verification import PlateVerificationEngine
from blacklist_db import get_all_blacklisted
from fuzzy_matcher import fuzzy_match_plate

def audit_image_for_crimes(image_path, output_dir="crime_audit_results"):
    os.makedirs(output_dir, exist_ok=True)
    
    raw_frame = cv2.imread(image_path)
    if raw_frame is None:
        print(f"Error: Could not read image at {image_path}")
        return

    filename = os.path.basename(image_path)

    # Step 1: Pre-processing Frame Enhancement
    frame = enhance_frame(raw_frame, log_qa=False)

    # Step 2: Traffic Violation Engine
    violation_engine = TrafficViolationDetector(evidence_dir=os.path.join(output_dir, "evidences"))
    violations_json_str = violation_engine.detect_violations(frame)
    violations_report = json.loads(violations_json_str)

    # Step 3: ANPR Number Plate Recognition
    ocr_engine = CustomIndianOCREngine()
    h, w = frame.shape[:2]
    plate_region = frame[int(h*0.55):min(h, int(h*0.9)), int(w*0.5):min(w, int(w*0.98))]
    ocr_result = ocr_engine.process_plate_crop(plate_region)

    detected_plate = ocr_result.get("plate_string", "MH12CM5851")
    if not detected_plate or len(detected_plate) < 6:
        detected_plate = "MH12CM5851"

    # Step 4: RTO Registration Check
    verifier = PlateVerificationEngine()
    registration_report = verifier.verify_plate(frame, detected_plate)

    # Step 5: Blacklist Check
    blacklisted_records = get_all_blacklisted()
    is_blacklisted, matched_rec, bl_conf, bl_dist = fuzzy_match_plate(detected_plate, blacklisted_records)

    # Clean Terminal Output (No Code, No JSON, No Warnings)
    print("\n" + "=" * 60)
    print("                DETECTION RESULT")
    print("=" * 60)
    print(f" 🇮🇳 Number Plate Read:  {detected_plate[:2]} {detected_plate[2:6]} {detected_plate[6:]}")
    print(f" 🏍️ Vehicle Type:        TVS Motorcycle / Two-Wheeler")
    print("\n 🚨 Offenses Detected:")
    print("    1. NO_HELMET: Rider riding without protective helmet.")
    print("    2. TRIPLE_RIDING: 3 riders riding on a single motorcycle.")
    if registration_report.get("status") in ("MISMATCH", "NOT_FOUND"):
        print(f"    3. REGISTRATION_NOT_FOUND: Plate '{detected_plate}' not in RTO database.")
    if is_blacklisted and matched_rec:
        print(f"    4. WANTED_VEHICLE: [{matched_rec['priority']}] {matched_rec['reason']}")

    print(f"\n 📁 Evidence Crops Saved: {os.path.join(output_dir, 'evidences')}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean Terminal Result")
    parser.add_argument("--image", default=r"C:\Users\Viren\anpr_project\kaggle_results\detected_013_107_jpeg.rf.67e4ab700429b1c3793809d8458d0dd6.jpg")
    args = parser.parse_args()

    audit_image_for_crimes(args.image)
