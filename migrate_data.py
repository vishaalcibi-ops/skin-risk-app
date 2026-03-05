import json
import os
from datetime import datetime
from app import app, db, User, Scan

def migrate():
    with app.app_context():
        # 1. Ensure a user exists to attach history to
        user = User.query.first()
        if not user:
            print("No users found. Creating a default account: admin@example.com / admin123")
            user = User(email='admin@example.com', name='Admin')
            user.set_password('admin123')
            db.session.add(user)
            db.session.commit()
        
        # 2. Read legacy JSON history
        HISTORY_FILE = 'scans_history.json'
        if not os.path.exists(HISTORY_FILE):
            print("No legacy history file found.")
            return
        
        try:
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
        except Exception as e:
            print(f"Error reading JSON history: {e}")
            return
        
        # 3. Migrate records
        count = 0
        for record in history:
            # Avoid duplicates (simple check by image_url)
            exists = Scan.query.filter_by(image_url=record['image_url'], user_id=user.id).first()
            if not exists:
                # Convert timestamp string to datetime object
                # Original format: "%Y-%m-%d %H:%M:%S"
                try:
                    ts = datetime.strptime(record['timestamp'], "%Y-%m-%d %H:%M:%S")
                except:
                    ts = datetime.utcnow()
                
                new_scan = Scan(
                    disease=record['disease'],
                    confidence=record['confidence'],
                    risk_level=record['risk_level'],
                    image_url=record['image_url'],
                    timestamp=ts,
                    user_id=user.id
                )
                db.session.add(new_scan)
                count += 1
        
        db.session.commit()
        print(f"Successfully migrated {count} scan records to user {user.email}.")

if __name__ == '__main__':
    migrate()
