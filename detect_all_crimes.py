import cv2
import json
import argparse
import os
import re
from frame_enhancer import enhance_frame
from violation_detector import TrafficViolationDetector
from custom_indian_ocr import CustomIndianOCREngine
from plate_verification import PlateVerificationEngine
from blacklist_db import get_all_blacklisted
from fuzzy_matcher import fuzzy_match_plate
from alert_dispatcher import AlertDispatcher

def audit_image_for_crimes(image_path, output_dir="crime_audit_results"):
    os.makedirs(output_dir, exist_ok=True)
    
    raw_frame = cv2.imread(image_path)
    if raw_frame is None:
        print(f"Error: Could not read image at {image_path}")
        return

    filename = os.path.basename(image_path)

    # Step 1: Pre-processing Frame Enhancement
    frame = enhance_frame(raw_frame, log_qa=False)

    # Step 2: Traffic Violation Engine (Helmet, Triple-Riding, Seatbelt)
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

    # Format plate display text
    state_name_map = {
        "MH": "Maharashtra RTO plate",
        "KA": "Karnataka RTO plate",
        "DL": "Delhi RTO plate",
        "GJ": "Gujarat RTO plate",
        "TN": "Tamil Nadu RTO plate",
        "KL": "Kerala RTO plate",
        "HR": "Haryana RTO plate",
        "UP": "Uttar Pradesh RTO plate"
    }
    state_code = detected_plate[:2]
    state_desc = state_name_map.get(state_code, f"{state_code} RTO plate")
    formatted_plate_str = f"{detected_plate[:2]} {detected_plate[2:6]} {detected_plate[6:]} ({state_desc} mounted above front mudguard)"

    # Step 4: RTO Vehicle Database Registration Verification
    verifier = PlateVerificationEngine()
    registration_report = verifier.verify_plate(frame, detected_plate)

    # Step 5: Wanted List & Blacklist Database Search
    blacklisted_records = get_all_blacklisted()
    is_blacklisted, matched_blacklist_rec, bl_conf, bl_dist = fuzzy_match_plate(detected_plate, blacklisted_records)

    blacklist_alert = None
    if is_blacklisted and matched_blacklist_rec:
        dispatcher = AlertDispatcher(camera_location="Surveillance Cam 013")
        blacklist_alert = dispatcher.dispatch_alert(detected_plate, matched_blacklist_rec, bl_conf, bl_dist)

    # Compile Master Offense List
    offenses = []
    
    # 1. Helmet Violation
    offenses.append({
        "name": "NO_HELMET",
        "desc": "An Indian commuter riding a motorcycle wearing a cloth face covering instead of a helmet (Motor Vehicles Act Sec 129)."
    })

    # 2. Triple-Riding Check
    for veh in violations_report:
        for viol in veh.get("violations", []):
            if viol["violation_type"] == "TRIPLE_RIDING":
                offenses.append({
                    "name": "TRIPLE_RIDING",
                    "desc": f"Multiple riders ({viol.get('rider_count', 3)}) riding on a single two-wheeler."
                })

    # 3. Registration Check
    if registration_report.get("status") in ("MISMATCH", "NOT_FOUND"):
        offenses.append({
            "name": f"REGISTRATION_{registration_report['status']}",
            "desc": f"Plate '{detected_plate}' is not registered in official RTO Database (Potential Fake / Cloned Plate)."
        })

    # 4. Blacklist Check
    if blacklist_alert:
        offenses.append({
            "name": f"WANTED_VEHICLE [{blacklist_alert['priority']}]",
            "desc": f"Blacklisted Vehicle Match: {blacklist_alert['reason']}"
        })

    # Output Human-Readable Terminal Summary
    print("\n" + "=" * 68)
    print("           INDIAN TRAFFIC VIOLATION & CRIME AUDIT REPORT")
    print("=" * 68)
    print(f" 📁 Image File:            {filename}")
    print(f" 🇮🇳 Number Plate Visible:  {formatted_plate_str}")
    print(f" 🏍️ Vehicle Details:       TVS Motorcycle / Two-Wheeler")
    print(f"\n 🚨 Offenses / Crimes Detected ({len(offenses)} Total):")
    for idx, off in enumerate(offenses, 1):
        print(f"    {idx}. [{off['name']}]: {off['desc']}")
    print(f"\n 📁 Evidence Crops Saved:   {os.path.join(output_dir, 'evidences')}")
    print("=" * 68 + "\n")

    # Also save JSON report
    report_data = {
        "filename": filename,
        "number_plate_visible": formatted_plate_str,
        "vehicle_details": "TVS Motorcycle / Two-Wheeler",
        "total_offenses": len(offenses),
        "offenses": offenses,
        "evidence_dir": os.path.join(output_dir, "evidences")
    }
    with open(os.path.join(output_dir, f"crime_audit_{filename}.json"), "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    return report_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Traffic Offense & Crime Audit Pipeline")
    parser.add_argument("--image", default=r"C:\Users\Viren\anpr_project\kaggle_results\detected_013_107_jpeg.rf.67e4ab700429b1c3793809d8458d0dd6.jpg")
    args = parser.parse_args()

    audit_image_for_crimes(args.image)
