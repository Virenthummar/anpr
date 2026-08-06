import cv2
import argparse
import json
from violation_detector import TrafficViolationDetector

def main():
    parser = argparse.ArgumentParser(description="Test Traffic Violation Detection (Helmet, Triple-Riding, Seatbelt)")
    parser.add_argument("--image", default="num2.png", help="Path to input image file")
    parser.add_argument("--helmet-model", help="Optional path to custom helmet model checkpoint")
    parser.add_argument("--seatbelt-model", help="Optional path to custom seatbelt model checkpoint")
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Error: Could not read image {args.image}")
        return

    print(f"\n==================================================")
    print(f"Running Traffic Violation Detector on: {args.image}")
    print(f"==================================================")

    detector = TrafficViolationDetector(
        evidence_dir="evidences",
        helmet_model_path=args.helmet_model,
        seatbelt_model_path=args.seatbelt_model
    )

    # Example ANPR pre-detected plates format (optional)
    anpr_plates = [
        {"plate_number": "KA03NA5278", "bbox": (30, 200, 150, 250)},
        {"plate_number": "DL10CC8821", "bbox": (700, 200, 850, 250)},
        {"plate_number": "GJ01DA1234", "bbox": (750, 500, 880, 550)}
    ]

    report_json = detector.detect_violations(frame, anpr_vehicles=anpr_plates)

    print("\n--- VIOLATION REPORT JSON OUTPUT ---")
    print(report_json)

    parsed = json.loads(report_json)
    print(f"\nTotal Vehicles Processed: {len(parsed)}")
    
    total_violations = sum(len(v["violations"]) for v in parsed)
    print(f"Total Traffic Violations Flagged: {total_violations}")

if __name__ == "__main__":
    main()
