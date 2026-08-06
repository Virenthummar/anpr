import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "vehicle_registry.db")

def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS registrations (
            plate_number TEXT PRIMARY KEY,
            make TEXT NOT NULL,
            model TEXT NOT NULL,
            color TEXT NOT NULL,
            vehicle_type TEXT NOT NULL,
            registered_state TEXT NOT NULL,
            owner_name TEXT
        )
    """)

    # Seed initial mock data for testing
    mock_records = [
        ("KA03NA5278", "Maruti Suzuki", "Ciaz", "Grey", "Sedan", "KA", "Rahul Sharma"),
        ("DL10CC8821", "Honda", "City", "Grey", "Sedan", "DL", "Anish Verma"),
        ("GJ01DA1234", "Tata", "Nexon EV", "Blue", "SUV", "GJ", "Patel Enterprise"),
        ("TN01BX1045", "Toyota", "Yaris", "Silver", "Sedan", "TN", "Srinivasan K"),
        ("MH12CD5678", "Hyundai", "Creta", "White", "SUV", "MH", "Priya Kulkarni")
    ]

    for record in mock_records:
        cursor.execute("""
            INSERT OR REPLACE INTO registrations (plate_number, make, model, color, vehicle_type, registered_state, owner_name)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, record)

    conn.commit()
    conn.close()
    print(f"[VehicleDB] Initialized database with {len(mock_records)} registration records at {db_path}")

def get_registration(plate_number, db_path=DB_PATH):
    if not os.path.exists(db_path):
        init_db(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT plate_number, make, model, color, vehicle_type, registered_state, owner_name FROM registrations WHERE plate_number = ?", (plate_number.upper(),))
    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            "plate_number": row[0],
            "make": row[1],
            "model": row[2],
            "color": row[3],
            "vehicle_type": row[4],
            "registered_state": row[5],
            "owner_name": row[6]
        }
    return None

if __name__ == "__main__":
    init_db()
