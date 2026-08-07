import os
import glob
import cv2
import json
from violation_detector import TrafficViolationDetector
from anpr_image import process_image

DATASET_VAL_DIR = r"C:\Users\Viren\.cache\kagglehub\datasets\aryanvaid13\indian-helmet-detection-dataset\versions\1\valid\images"
OUTPUT_DIR = r"C:\Users\Viren\anpr_project\kaggle_results"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    detector = TrafficViolationDetector(evidence_dir=os.path.join(OUTPUT_DIR, "evidences"))

    image_paths = glob.glob(os.path.join(DATASET_VAL_DIR, "*.jpg"))[:5]
    print(f"\n============================================================")
    print(f"   RUNNING TRAFFIC VIOLATION ENGINE ON KAGGLE HELMET DATASET")
    print(f"   Dataset Source: aryanvaid13/indian-helmet-detection-dataset")
    print(f"============================================================")
    print(f"Found {len(image_paths)} sample validation images to process.")

    results_summary = []

    for idx, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        print(f"\n---> Processing Image [{idx+1}/{len(image_paths)}]: {filename}")
        
        # Run violation detector
        report_json_str = detector.detect_violations(frame)
        report = json.loads(report_json_str)

        out_img_path = os.path.join(OUTPUT_DIR, f"detected_{filename}")
        cv2.imwrite(out_img_path, frame)

        summary_item = {
            "image_filename": filename,
            "saved_result_image": out_img_path,
            "detected_vehicles": len(report),
            "report": report
        }
        results_summary.append(summary_item)

    summary_file = os.path.join(OUTPUT_DIR, "kaggle_helmet_detection_report.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    print("\n" + "=" * 60)
    print("                 DETECTION SUMMARY REPORT")
    print("=" * 60)
    print(json.dumps(results_summary, indent=2))
    print(f"\nSummary JSON saved to: {summary_file}")
    print(f"Annotated result images saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
