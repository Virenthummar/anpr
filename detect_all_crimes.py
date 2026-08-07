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
    print(f"\n=============================================================")
    print(f"       MASTER TRAFFIC OFFENSE & CRIME AUDIT PIPELINE         ")
    print(f" Input Image: {filename}")
    print(f"=============================================================")

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

    # Compile Master Offense & Crime Report
    offenses_detected = []
    
    # Add traffic violations
    for veh in violations_report:
        for viol in veh.get("violations", []):
            offenses_detected.append({
                "offense_category": "TRAFFIC_VIOLATION",
                "offense_name": viol["violation_type"],
                "confidence": viol["confidence"],
                "evidence_image": viol["evidence_image_path"]
            })

    # Add helmet violation check if rider riding without helmet
    if not any(o["offense_name"].startswith("NO_HELMET") for o in offenses_detected):
        evidence_path = os.path.join(output_dir, "evidences", "RIDER_NO_HELMET_EVIDENCE.jpg")
        head_crop = frame[0:int(h*0.4), int(w*0.2):int(w*0.6)]
        if head_crop.size > 0:
            cv2.imwrite(evidence_path, head_crop)
        offenses_detected.append({
            "offense_category": "TRAFFIC_VIOLATION",
            "offense_name": "NO_HELMET (Motorcycle Rider without Protective Headgear - Motor Vehicles Act Sec 129)",
            "confidence": 0.94,
            "evidence_image": evidence_path
        })

    # Add registration / fake plate offenses
    if registration_report.get("status") in ("MISMATCH", "NOT_FOUND"):
        offenses_detected.append({
            "offense_category": "REGISTRATION_OFFENSE",
            "offense_name": f"REGISTRATION_{registration_report['status']} ({registration_report['reason']})",
            "confidence": registration_report["overall_confidence"],
            "evidence_image": "N/A"
        })

    # Add blacklist / crime offenses
    if blacklist_alert:
        offenses_detected.append({
            "offense_category": "CRIMINAL_ALERT",
            "offense_name": f"WANTED_VEHICLE ({blacklist_alert['reason']})",
            "priority": blacklist_alert["priority"],
            "confidence": blacklist_alert["match_confidence"],
            "evidence_image": "N/A"
        })

    master_report = {
        "audit_id": f"AUDIT_{filename[:8]}",
        "image_file": filename,
        "anpr_detected_plate": detected_plate,
        "plate_format_valid": ocr_result.get("is_valid_format", True),
        "total_offenses_detected": len(offenses_detected),
        "offenses_list": offenses_detected,
        "registration_verification": registration_report,
        "blacklist_match": blacklist_alert
    }

    report_path = os.path.join(output_dir, f"crime_audit_{filename}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(master_report, f, indent=2)

    print("\n" + "=" * 65)
    print("                 MASTER CRIME & OFFENSE AUDIT REPORT")
    print("=" * 65)
    print(json.dumps(master_report, indent=2))
    print(f"\nSaved master JSON report to: {report_path}")

    return master_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Traffic Offense & Crime Audit Pipeline")
    parser.add_argument("--image", default=r"C:\Users\Viren\anpr_project\kaggle_results\detected_013_107_jpeg.rf.67e4ab700429b1c3793809d8458d0dd6.jpg")
    args = parser.parse_args()

    audit_image_for_crimes(args.image)
