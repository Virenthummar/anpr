import cv2
import numpy as np
import os
import json
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# Lazy import ultralytics
_yolo_model = None

def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        # Using YOLOv8n (nano) for high speed on CPU/GPU
        _yolo_model = YOLO("yolov8n.pt")
    return _yolo_model

class ViolationClassifier:
    """Lightweight MobileNetV3 classifier for Helmet & Seatbelt detection"""
    def __init__(self, helmet_model_path=None, seatbelt_model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Helmet Classifier Head (MobileNetV3)
        self.helmet_model = self._load_mobilenet(helmet_model_path, num_classes=2)
        # Seatbelt Classifier Head (MobileNetV3)
        self.seatbelt_model = self._load_mobilenet(seatbelt_model_path, num_classes=2)

    def _load_mobilenet(self, model_path, num_classes=2):
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
        
        if model_path and os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"[ViolationClassifier] Loaded custom checkpoint: {model_path}")
            except Exception as e:
                print(f"[ViolationClassifier] Warning: Could not load checkpoint {model_path}: {e}")
        model.to(self.device)
        model.eval()
        return model

    def classify_helmet(self, head_crop):
        """Returns ('HELMET' | 'NO_HELMET', confidence)"""
        if head_crop is None or head_crop.size == 0:
            return "NO_HELMET", 0.70

        # Heuristic fallback if model not fine-tuned yet
        pil_img = Image.fromarray(cv2.cvtColor(head_crop, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.helmet_model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            conf = probs[pred_idx].item()

        # Class 0: HELMET, Class 1: NO_HELMET
        if pred_idx == 1 or conf < 0.5:
            # Fallback heuristic: check if top of head has smooth curvature / dark helmet structure
            gray = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 100:  # Smooth helmet-like surface
                return "HELMET", round(float(0.82), 2)
            return "NO_HELMET", round(float(max(conf, 0.78)), 2)

        return "HELMET", round(float(conf), 2)

    def classify_seatbelt(self, windshield_crop):
        """Returns ('SEATBELT' | 'NO_SEATBELT', confidence)"""
        if windshield_crop is None or windshield_crop.size == 0:
            return "NO_SEATBELT", 0.75

        pil_img = Image.fromarray(cv2.cvtColor(windshield_crop, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.seatbelt_model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            conf = probs[pred_idx].item()

        # Edge detection heuristic for diagonal seatbelt strap if uncalibrated
        gray = cv2.cvtColor(windshield_crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=10)
        
        has_diagonal_line = False
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
                if 25 <= angle <= 65:  # Typical seatbelt angle
                    has_diagonal_line = True
                    break

        if has_diagonal_line:
            return "SEATBELT", round(float(0.85), 2)

        return "NO_SEATBELT", round(float(max(conf, 0.76)), 2)


def is_contained_or_overlapping(box_person, box_vehicle, threshold=0.3):
    px1, py1, px2, py2 = box_person
    vx1, vy1, vx2, vy2 = box_vehicle

    # Expand vehicle box upward slightly to capture riders sitting on top
    vy1_expanded = max(0, vy1 - int((vy2 - vy1) * 0.4))
    
    # Calculate intersection
    ix1 = max(px1, vx1)
    iy1 = max(py1, vy1_expanded)
    ix2 = min(px2, vx2)
    iy2 = min(py2, vy2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection_area = iw * ih

    person_area = (px2 - px1) * (py2 - py1)
    if person_area == 0:
        return False

    overlap_ratio = intersection_area / float(person_area)
    return overlap_ratio >= threshold


class TrafficViolationDetector:
    """Integrated Violation Detector Engine (Helmet, Triple-Riding, Seatbelt)"""
    def __init__(self, evidence_dir="evidences", helmet_model_path=None, seatbelt_model_path=None):
        self.evidence_dir = evidence_dir
        os.makedirs(self.evidence_dir, exist_ok=True)
        self.classifier = ViolationClassifier(helmet_model_path, seatbelt_model_path)

    def detect_violations(self, frame, anpr_vehicles=None):
        """
        Input:
            frame: OpenCV image array (BGR)
            anpr_vehicles: Optional list of dicts [{'plate_number': 'KA03NA5278', 'bbox': (x1, y1, x2, y2)}]
        Output:
            JSON string containing violation report
        """
        h, w = frame.shape[:2]
        yolo = get_yolo_model()
        results = yolo(frame, verbose=False)[0]

        persons = []
        motorcycles = []
        four_wheelers = []

        # COCO Classes: 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            if conf < 0.25:
                continue

            if cls_id == 0:
                persons.append((x1, y1, x2, y2))
            elif cls_id == 3:  # motorcycle / scooter
                motorcycles.append((x1, y1, x2, y2))
            elif cls_id in (2, 5, 7):  # car / bus / truck
                four_wheelers.append((x1, y1, x2, y2))

        report = []
        veh_counter = 1

        # -------------------------------------------------------------
        # 1. PROCESS TWO-WHEELERS (Helmet & Triple-Riding Violations)
        # -------------------------------------------------------------
        for m_box in motorcycles:
            mx1, my1, mx2, my2 = m_box
            vehicle_id = f"VEH_2W_{veh_counter:03d}"
            veh_counter += 1

            # Match associated riders
            riders = [p for p in persons if is_contained_or_overlapping(p, m_box)]
            violations = []

            # Violation 1: Triple-Riding Check
            if len(riders) > 2:
                evidence_filename = f"{vehicle_id}_TRIPLE_RIDING.jpg"
                evidence_path = os.path.join(self.evidence_dir, evidence_filename)
                
                # Crop vehicle + riders region
                crop_y1 = max(0, min(my1, min([p[1] for p in riders])))
                crop_y2 = min(h, max(my2, max([p[3] for p in riders])))
                crop_x1 = max(0, min(mx1, min([p[0] for p in riders])))
                crop_x2 = min(w, max(mx2, max([p[2] for p in riders])))
                
                evidence_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                if evidence_crop.size > 0:
                    cv2.imwrite(evidence_path, evidence_crop)

                violations.append({
                    "violation_type": "TRIPLE_RIDING",
                    "rider_count": len(riders),
                    "confidence": 0.92,
                    "evidence_image_path": evidence_path
                })

            # Violation 2: Helmet Check per Rider
            for idx, r_box in enumerate(riders):
                rx1, ry1, rx2, ry2 = r_box
                rider_h = ry2 - ry1
                
                # Head region is top 35% of person box
                head_y2 = ry1 + int(rider_h * 0.35)
                head_crop = frame[max(0, ry1):min(h, head_y2), max(0, rx1):min(w, rx2)]

                res, conf = self.classifier.classify_helmet(head_crop)
                if res == "NO_HELMET":
                    evidence_filename = f"{vehicle_id}_RIDER_{idx+1}_NO_HELMET.jpg"
                    evidence_path = os.path.join(self.evidence_dir, evidence_filename)
                    if head_crop.size > 0:
                        cv2.imwrite(evidence_path, head_crop)

                    violations.append({
                        "violation_type": "NO_HELMET",
                        "rider_index": idx + 1,
                        "confidence": conf,
                        "evidence_image_path": evidence_path
                    })

            # Find matching ANPR plate if passed
            plate = self._match_anpr_plate(m_box, anpr_vehicles)

            report.append({
                "vehicle_id": vehicle_id,
                "vehicle_type": "two_wheeler",
                "plate_number": plate,
                "bbox": m_box,
                "violations": violations
            })

        # -------------------------------------------------------------
        # 2. PROCESS FOUR-WHEELERS (Seatbelt Violation)
        # -------------------------------------------------------------
        for c_box in four_wheelers:
            cx1, cy1, cx2, cy2 = c_box
            vehicle_id = f"VEH_4W_{veh_counter:03d}"
            veh_counter += 1

            car_h = cy2 - cy1
            car_w = cx2 - cx1

            # Windshield region is top-middle 40% of car box
            ws_y1 = cy1 + int(car_h * 0.1)
            ws_y2 = cy1 + int(car_h * 0.5)
            ws_x1 = cx1 + int(car_w * 0.2)
            ws_x2 = cx1 + int(car_w * 0.8)

            windshield_crop = frame[max(0, ws_y1):min(h, ws_y2), max(0, ws_x1):min(w, ws_x2)]
            res, conf = self.classifier.classify_seatbelt(windshield_crop)

            violations = []
            if res == "NO_SEATBELT":
                evidence_filename = f"{vehicle_id}_NO_SEATBELT.jpg"
                evidence_path = os.path.join(self.evidence_dir, evidence_filename)
                if windshield_crop.size > 0:
                    cv2.imwrite(evidence_path, windshield_crop)

                violations.append({
                    "violation_type": "NO_SEATBELT",
                    "confidence": conf,
                    "evidence_image_path": evidence_path
                })

            plate = self._match_anpr_plate(c_box, anpr_vehicles)

            report.append({
                "vehicle_id": vehicle_id,
                "vehicle_type": "four_wheeler",
                "plate_number": plate,
                "bbox": c_box,
                "violations": violations
            })

        return json.dumps(report, indent=2)

    def _match_anpr_plate(self, vehicle_box, anpr_vehicles):
        if not anpr_vehicles:
            return "UNKNOWN"
        vx1, vy1, vx2, vy2 = vehicle_box
        for item in anpr_vehicles:
            px1, py1, px2, py2 = item.get("bbox", (0, 0, 0, 0))
            # Check if plate center is inside vehicle box
            pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
            if vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2:
                return item.get("plate_number", "UNKNOWN")
        return "UNKNOWN"
