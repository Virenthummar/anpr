import cv2
import json
import argparse
from custom_indian_ocr import CustomIndianOCREngine

def main():
    parser = argparse.ArgumentParser(description="Test Custom Indian CRNN OCR Engine with Deskewing & Layout Handling")
    parser.add_argument("--image", default="num.png", help="Path to input vehicle image file")
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Error: Could not read image {args.image}")
        return

    ocr_engine = CustomIndianOCREngine()

    print("\n" + "=" * 60)
    print("      RUNNING CUSTOM INDIAN CRNN OCR PIPELINE TEST")
    print("=" * 60)

    # Detect plate crops using Haar Cascade
    cascade_path = "haarcascade_russian_plate_number.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(30, 10))

    if len(plates) == 0:
        print("Haar Cascade found no bounding boxes, running OCR on central crop...")
        h, w = frame.shape[:2]
        crop = frame[int(h*0.3):int(h*0.8), int(w*0.2):int(w*0.8)]
        res = ocr_engine.process_plate_crop(crop)
        print(json.dumps(res, indent=2))
    else:
        for idx, (x, y, w, h) in enumerate(plates):
            crop = frame[y:y+h, x:x+w]
            res = ocr_engine.process_plate_crop(crop)
            print(f"\n--- Candidate Crop {idx+1} ---")
            print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
