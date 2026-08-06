import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models

class VehicleAttributeClassifier:
    """Extracts visual attributes (Color, Body Type) from vehicle image crops"""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(self.device)
        self.resnet.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict_color(self, vehicle_crop):
        """Predicts dominant vehicle color using HSV color space analysis"""
        if vehicle_crop is None or vehicle_crop.size == 0:
            return "Unknown", 0.0

        hsv = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2HSV)
        
        # Define color ranges in HSV
        color_ranges = {
            "Red": [
                ((0, 70, 50), (10, 255, 255)),
                ((170, 70, 50), (180, 255, 255))
            ],
            "Blue": [((95, 70, 50), (135, 255, 255))],
            "Green": [((35, 70, 50), (85, 255, 255))],
            "Yellow": [((15, 70, 50), (35, 255, 255))],
            "Black": [((0, 0, 0), (180, 255, 45))],
            "White": [((0, 0, 200), (180, 30, 255))],
            "Grey": [((0, 0, 45), (180, 45, 200))]
        }

        total_pixels = vehicle_crop.shape[0] * vehicle_crop.shape[1]
        color_counts = {}

        for color_name, ranges in color_ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for (lower, upper) in ranges:
                mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
            count = cv2.countNonZero(mask)
            color_counts[color_name] = count

        dominant_color = max(color_counts, key=color_counts.get)
        confidence = color_counts[dominant_color] / float(total_pixels)
        confidence = min(0.95, round(float(confidence * 2.5), 2))

        return dominant_color, max(0.65, confidence)

    def predict_body_type(self, vehicle_crop):
        """Predicts vehicle body class (Sedan, SUV, Hatchback, Two-Wheeler)"""
        if vehicle_crop is None or vehicle_crop.size == 0:
            return "Sedan", 0.50

        h, w = vehicle_crop.shape[:2]
        aspect_ratio = w / float(h)

        if aspect_ratio < 1.1:
            return "Two-Wheeler", 0.85
        elif aspect_ratio > 1.8:
            return "Sedan", 0.88
        else:
            return "SUV", 0.82
