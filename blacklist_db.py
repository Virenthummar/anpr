import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "blacklist.db")

def init_blacklist_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS blacklist (
            plate_number TEXT PRIMARY KEY,
            reason TEXT NOT NULL,
            priority TEXT NOT NULL CHECK(priority IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')),
            added_by TEXT DEFAULT 'SYSTEM',
            added_timestamp TEXT NOT NULL,
            notes TEXT
        )
    """)

    # Seed mock blacklisted plates
    mock_blacklist = [
        ("KA03NA5278", "Stolen Vehicle - FIR 104/2026", "CRITICAL", "Traffic Police HQ", "2026-08-01 10:00:00", "Red Flag: Armed suspects"),
        ("DL10CC8821", "Expired Registration & Unpaid Fines", "MEDIUM", "RTO Delhi", "2026-08-02 11:30:00", "Vehicle impound order"),
        ("GJ01DA1234", "Wanted Criminal Vehicle", "CRITICAL", "CID Gujarat", "2026-08-03 14:15:00", "Intercept immediately"),
        ("TN01BX1045", "Suspected Hit and Run", "HIGH", "Chennai City Police", "2026-08-04 09:45:00", "Witness reported MB crash")
    ]

    for record in mock_blacklist:
        cursor.execute("""
            INSERT OR REPLACE INTO blacklist (plate_number, reason, priority, added_by, added_timestamp, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, record)

    conn.commit()
    conn.close()
    print(f"[BlacklistDB] Initialized database with {len(mock_blacklist)} blacklisted records at {db_path}")

def add_to_blacklist(plate_number, reason, priority="HIGH", added_by="USER", notes="", db_path=DB_PATH):
    if not os.path.exists(db_path):
        init_blacklist_db(db_path)

    plate_clean = plate_number.upper().strip()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO blacklist (plate_number, reason, priority, added_by, added_timestamp, notes)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (plate_clean, reason, priority, added_by, now_str, notes))
    conn.commit()
    conn.close()
    return {"plate_number": plate_clean, "reason": reason, "priority": priority, "added_timestamp": now_str}

def remove_from_blacklist(plate_number, db_path=DB_PATH):
    if not os.path.exists(db_path):
        return False

    plate_clean = plate_number.upper().strip()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM blacklist WHERE plate_number = ?", (plate_clean,))
    rows = cursor.rowcount
    conn.commit()
    conn.close()
    return rows > 0

def get_all_blacklisted(db_path=DB_PATH):
    if not os.path.exists(db_path):
        init_blacklist_db(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT plate_number, reason, priority, added_by, added_timestamp, notes FROM blacklist")
    rows = cursor.fetchall()
    conn.close()

    result = []
    for r in rows:
        result.append({
            "plate_number": r[0],
            "reason": r[1],
            "priority": r[2],
            "added_by": r[3],
            "added_timestamp": r[4],
            "notes": r[5]
        })
    return result

if __name__ == "__main__":
    init_blacklist_db()
