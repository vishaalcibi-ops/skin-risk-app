import os
from app import app, db, Scan, User

def diagnose():
    with app.app_context():
        print("Checking Scan records...")
        scans = Scan.query.all()
        for s in scans:
            try:
                print(f"Scan ID: {s.id}, User ID: {s.user_id}, Disease: {s.disease}")
                if s.timestamp:
                    print(f"  Timestamp: {s.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    print("  [ERROR] Timestamp is None")
                
                if s.confidence is None:
                    print("  [ERROR] Confidence is None")
                
                if s.risk_level is None:
                    print("  [ERROR] Risk Level is None")
                    
            except Exception as e:
                print(f"  [CRITICAL ERROR] Failed to access scan {s.id}: {e}")

        print("\nChecking User records...")
        users = User.query.all()
        for u in users:
            print(f"User ID: {u.id}, Email: {u.email}")

if __name__ == "__main__":
    diagnose()
