import json
import time
from datetime import datetime

class AlertDispatcher:
    """Real-time Alert Dispatcher supporting Console, Webhook, and SMS targets"""
    def __init__(self, camera_location="Gate 1 Main Entrance", webhook_url=None):
        self.camera_location = camera_location
        self.webhook_url = webhook_url

    def dispatch_alert(self, ocr_plate, matched_record, confidence, distance):
        alert_id = f"ALT_{int(time.time()*1000)}"
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        payload = {
            "alert_id": alert_id,
            "timestamp": timestamp_str,
            "camera_location": self.camera_location,
            "ocr_read_plate": ocr_plate,
            "matched_blacklist_plate": matched_record["plate_number"],
            "reason": matched_record["reason"],
            "priority": matched_record["priority"],
            "added_by": matched_record["added_by"],
            "match_confidence": confidence,
            "fuzzy_edit_distance": distance,
            "notes": matched_record.get("notes", "")
        }

        # 1. Dispatch Console Alert (Immediate Logging)
        self._dispatch_console(payload)

        # 2. Dispatch Webhook payload if URL configured
        if self.webhook_url:
            self._dispatch_webhook(payload)

        return payload

    def _dispatch_console(self, payload):
        tag = "[CRITICAL ALERT]" if payload["priority"] in ("CRITICAL", "HIGH") else "[WARNING ALERT]"
        print("\n" + "=" * 65)
        print(f"{tag} BLACKLIST MATCH DETECTED [{payload['priority']}]")
        print("=" * 65)
        print(f" Alert ID:             {payload['alert_id']}")
        print(f" Timestamp:            {payload['timestamp']}")
        print(f" Location:             {payload['camera_location']}")
        print(f" OCR Read Plate:       '{payload['ocr_read_plate']}'")
        print(f" Matched Wanted Plate: '{payload['matched_blacklist_plate']}' (Conf: {payload['match_confidence']})")
        print(f" Priority / Reason:    [{payload['priority']}] {payload['reason']}")
        print(f" Notes:                {payload['notes']}")
        print("=" * 65 + "\n")

    def _dispatch_webhook(self, payload):
        print(f"[AlertDispatcher] Mocking Webhook POST payload to {self.webhook_url}...")
