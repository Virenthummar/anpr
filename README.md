# Automatic Number Plate Recognition (ANPR) & Indian Traffic Violation System

An end-to-end Computer Vision & Deep Learning pipeline for detecting Indian Vehicle Registration Number Plates and automated Traffic Violation Detection (Helmet, Triple-Riding, and Seatbelt violations).

---

## 🚀 Features

### 🇮🇳 1. Indian ANPR Engine (`anpr.py` & `anpr_image.py`)
- **Plate Detection:** Hybrid detector using Haar Cascades + CRAFT OCR direct text scanning.
- **Indian State Code Filtering:** Validates plates against all 36 Indian States & UTs (`KA`, `MH`, `DL`, `GJ`, `TN`, etc.).
- **Auto-Correction Heuristics:** Automatically resolves common OCR character confusions (e.g., `0` vs `O`, `1` vs `I`, `8` vs `B`).
- **Strict Format Regex:** Enforces standard Indian format `[State][RTO][Series][4 Digits]` (e.g. `KA03NA5278`).

### 🚦 2. Traffic Violation Detector (`violation_detector.py`)
- **YOLOv8 Object Detection:** Detects motorcycles, cars, buses, trucks, and pedestrians.
- **Helmet Violation Detection:** Crops rider head region and classifies `HELMET` vs `NO_HELMET`.
- **Triple-Riding Detection:** Measures spatial overlap between riders and 2-wheelers. Flags `TRIPLE_RIDING` if rider count > 2.
- **Seatbelt Detection:** Crops driver-side windshield region and classifies `SEATBELT` vs `NO_SEATBELT`.
- **Evidence Snippets:** Automatically saves cropped violation evidence images to `evidences/`.
- **JSON Report:** Outputs structured JSON report linking plate numbers to vehicle violation records.

### 🎓 3. MobileNetV3 Fine-Tuning (`train_violation_classifier.py`)
- PyTorch script to train/fine-tune MobileNetV3 classifiers on custom Indian traffic datasets.

---

## 🛠️ Installation & Setup

```bash
git clone https://github.com/Virenthummar/anpr.git
cd anpr
pip install opencv-python-headless easyocr ultralytics torch torchvision numpy
```

---

## 💻 Usage

### Run ANPR on an Image:
```bash
python anpr_image.py --image num2.png
```

### Run ANPR on Video Footage:
```bash
python anpr.py --video input_traffic.mp4 --output result.mp4 --csv detections.csv
```

### Run Violation Detection Pipeline:
```bash
python test_violation_pipeline.py --image num2.png
```

### Fine-Tune Classifier on Custom Dataset:
```bash
python train_violation_classifier.py --data-dir dataset/ --task helmet --epochs 15
```
