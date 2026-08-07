import os
import glob
import cv2
import json
import numpy as np
from attribute_classifier import VehicleAttributeClassifier
from plate_verification import PlateVerificationEngine
from detect_all_crimes import audit_image_for_crimes

def main():
    print("\n============================================================")
    print("   VEHICLE MAKE, MODEL & RECOGNITION (VMMR) AUDIT PIPELINE")
    print("   Dataset Source: prabashwara/vmmrdb-dataset (Kaggle)")
    print("============================================================")

    cache_dir = r"C:\Users\Viren\.cache\kagglehub\datasets\prabashwara\vmmrdb-dataset\versions\1"
    
    image_files = glob.glob(os.path.join(cache_dir, "**", "*.jpg"), recursive=True)[:5]
    if not image_files:
        image_files = glob.glob(os.path.join(cache_dir, "**", "*.png"), recursive=True)[:5]

    if not image_files:
        image_files = [r"C:\Users\Viren\anpr_project\num2.png"]

    classifier = VehicleAttributeClassifier()
    verifier = PlateVerificationEngine()

    for idx, img_path in enumerate(image_files, 1):
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        filename = os.path.basename(img_path)
        color = classifier.predict_color(frame)
        body = classifier.predict_body_type(frame)

        print(f"\n------------------------------------------------------------")
        print(f" [VEHICLE IMAGE {idx}]: {filename}")
        print(f"   Visual Color:       {color}")
        print(f"   Body Classifier:    {body}")
        print(f"------------------------------------------------------------")
        
        audit_image_for_crimes(img_path)

if __name__ == "__main__":
    main()
