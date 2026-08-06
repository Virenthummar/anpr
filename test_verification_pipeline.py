import cv2
import json
import argparse
from plate_verification import PlateVerificationEngine

def main():
    parser = argparse.ArgumentParser(description="Test Vehicle Plate Database Verification & Fake Plate Detection")
    parser.add_argument("--image", default="num.png", help="Path to input vehicle image file")
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Error: Could not read image {args.image}")
        return

    verifier = PlateVerificationEngine()

    test_cases = [
        ("KA03NA5278", "Valid Registered Plate"),
        ("FAKE99XX99", "Unregistered / Fake Plate"),
        ("MH12CD5678", "Cloned Plate Mismatch Test")
    ]

    print("\n" + "=" * 60)
    print("      RUNNING VEHICLE PLATE REGISTRATION VERIFICATION TEST")
    print("=" * 60)

    for plate, description in test_cases:
        print(f"\n---> Testing: '{plate}' ({description})")
        report = verifier.verify_plate(frame, plate)
        print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
