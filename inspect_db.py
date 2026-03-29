import sqlite3
import os

db_path = r'c:\Users\ADMIN\Desktop\skindiseases\instance\skin_disease.db'

def check_db():
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print("Tables in database:")
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT count(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"- {table_name}: {count} records")

            # Show first 5 records for important tables
            if table_name in ['user', 'scan', 'chat_message']:
                print(f"  All records from {table_name}:")
                cursor.execute(f"SELECT * FROM {table_name}")
                rows = cursor.fetchall()
                for row in rows:
                    print(f"    {row}")
        
        conn.close()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    check_db()
