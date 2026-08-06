import json
import os
from vehicle_db import get_registration
from attribute_classifier import VehicleAttributeClassifier

class PlateVerificationEngine:
    """Cross-checks ANPR detected number plates against RTO database and visual attributes"""
    def __init__(self):
        self.attr_classifier = VehicleAttributeClassifier()

    def verify_plate(self, vehicle_crop, detected_plate):
        """
        Input:
            vehicle_crop: BGR numpy image array of vehicle
            detected_plate: OCR read plate string (e.g. 'KA03NA5278')
        Returns:
            Dict containing verification status, confidence, and mismatch details
        """
        # 1. Predict visual attributes from image crop
        pred_color, color_conf = self.attr_classifier.predict_color(vehicle_crop)
        pred_body, body_conf = self.attr_classifier.predict_body_type(vehicle_crop)

        # 2. Look up plate in database
        reg_info = get_registration(detected_plate)

        if reg_info is None:
            return {
                "plate_number": detected_plate,
                "status": "NOT_FOUND",
                "overall_confidence": 0.95,
                "flagged_for_human_review": True,
                "reason": f"Plate number '{detected_plate}' is not registered in database (Potential Fake / Cloned Plate).",
                "detected_visual_attributes": {
                    "predicted_color": pred_color,
                    "predicted_body_type": pred_body
                },
                "registered_info": None,
                "mismatches": []
            }

        # 3. Compare visual attributes against registered records
        mismatches = []
        
        # Color match check (Grey & Silver treated as equivalent)
        reg_color = reg_info["color"]
        color_equivalent = (reg_color.lower() in ("grey", "silver") and pred_color.lower() in ("grey", "silver"))
        
        if not color_equivalent and reg_color.lower() != pred_color.lower():
            mismatches.append({
                "field": "color",
                "registered_value": reg_color,
                "detected_value": pred_color,
                "confidence": color_conf
            })

        # Body type match check
        reg_type = reg_info["vehicle_type"]
        if reg_type.lower() != pred_body.lower():
            mismatches.append({
                "field": "vehicle_type",
                "registered_value": reg_type,
                "detected_value": pred_body,
                "confidence": body_conf
            })

        # Determine overall verification status
        if len(mismatches) == 0:
            status = "MATCH"
            flagged = False
            reason = "Plate registration verified. Visual attributes match database records."
            overall_confidence = round((color_conf + body_conf) / 2.0, 2)
        else:
            status = "MISMATCH"
            flagged = True
            mismatched_fields = ", ".join([m["field"] for m in mismatches])
            reason = f"Visual attribute mismatch detected on field(s): [{mismatched_fields}]. Flagged for human review."
            overall_confidence = max([m["confidence"] for m in mismatches])

        return {
            "plate_number": detected_plate,
            "status": status,
            "overall_confidence": overall_confidence,
            "flagged_for_human_review": flagged,
            "reason": reason,
            "registered_info": reg_info,
            "detected_visual_attributes": {
                "predicted_color": pred_color,
                "predicted_body_type": pred_body
            },
            "mismatches": mismatches
        }
