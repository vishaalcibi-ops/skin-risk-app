import sqlite3
import os

db_path = r'c:\Users\ADMIN\Desktop\skindiseases\instance\skin_disease.db'

def check_columns():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(user);")
    columns = cursor.fetchall()
    print("Columns in 'user' table:")
    for col in columns:
        print(f"- {col[1]} ({col[2]})")
    conn.close()

if __name__ == "__main__":
    check_columns()
