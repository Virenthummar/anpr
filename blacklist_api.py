from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import os

from blacklist_db import (
    init_blacklist_db,
    add_to_blacklist,
    remove_from_blacklist,
    get_all_blacklisted
)
from fuzzy_matcher import fuzzy_match_plate
from alert_dispatcher import AlertDispatcher

app = FastAPI(
    title="ANPR Blacklist & Real-Time Alerting REST API",
    description="Manage blacklisted plates and perform real-time ANPR fuzzy match alerting.",
    version="1.0.0"
)

# Initialize Database on startup
init_blacklist_db()
dispatcher = AlertDispatcher(camera_location="API Gateway Gate 1")

class BlacklistCreateRequest(BaseModel):
    plate_number: str = Field(..., example="KA03NA5278", description="License plate string")
    reason: str = Field(..., example="Stolen Vehicle FIR 104/2026", description="Reason for blacklisting")
    priority: str = Field("HIGH", example="CRITICAL", description="Priority level: CRITICAL, HIGH, MEDIUM, LOW")
    added_by: Optional[str] = Field("ADMIN", example="Traffic Police HQ")
    notes: Optional[str] = Field("", example="Armed suspects")

class CheckPlateRequest(BaseModel):
    ocr_plate: str = Field(..., example="KA03NA527B", description="OCR read plate string to check")
    camera_location: Optional[str] = Field("Gate 1 Main Entrance", description="Camera location identifier")

@app.get("/")
def root():
    return {"message": "ANPR Blacklist & Real-Time Alerting API is running!"}

@app.get("/api/v1/blacklist")
def list_blacklist():
    """Retrieve all blacklisted plate records"""
    records = get_all_blacklisted()
    return {"total": len(records), "blacklist": records}

@app.post("/api/v1/blacklist")
def create_blacklist_entry(item: BlacklistCreateRequest):
    """Add a new plate to the blacklist"""
    res = add_to_blacklist(
        plate_number=item.plate_number,
        reason=item.reason,
        priority=item.priority,
        added_by=item.added_by or "ADMIN",
        notes=item.notes or ""
    )
    return {"status": "success", "data": res}

@app.delete("/api/v1/blacklist/{plate_number}")
def delete_blacklist_entry(plate_number: str):
    """Remove a plate from the blacklist"""
    success = remove_from_blacklist(plate_number)
    if not success:
        raise HTTPException(status_code=404, detail=f"Plate '{plate_number}' not found in blacklist.")
    return {"status": "success", "message": f"Plate '{plate_number}' removed from blacklist."}

@app.post("/api/v1/check-plate")
def check_plate_and_alert(req: CheckPlateRequest):
    """Check ANPR OCR plate against blacklist using Levenshtein fuzzy matching"""
    records = get_all_blacklisted()
    is_match, matched_rec, conf, dist = fuzzy_match_plate(req.ocr_plate, records)

    if is_match and matched_rec:
        disp = AlertDispatcher(camera_location=req.camera_location or "Gate 1")
        alert_payload = disp.dispatch_alert(req.ocr_plate, matched_rec, conf, dist)
        return {
            "alert_triggered": True,
            "match_status": "MATCH_FOUND",
            "alert": alert_payload
        }

    return {
        "alert_triggered": False,
        "match_status": "NO_MATCH",
        "ocr_plate": req.ocr_plate,
        "message": "Plate is clean. No blacklist match found."
    }
