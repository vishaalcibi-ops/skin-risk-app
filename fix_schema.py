import sqlite3
import os

db_path = r'c:\Users\ADMIN\Desktop\skindiseases\instance\skin_disease.db'

def fix_schema():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # List of columns to add
    new_columns = [
        ('age', 'INTEGER'),
        ('skin_type', 'VARCHAR(50)'),
        ('location', 'VARCHAR(100)'),
        ('bio', 'TEXT')
    ]
    
    for col_name, col_type in new_columns:
        try:
            print(f"Adding column {col_name} to 'user' table...")
            cursor.execute(f"ALTER TABLE user ADD COLUMN {col_name} {col_type};")
            print(f"Successfully added {col_name}.")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print(f"Column {col_name} already exists.")
            else:
                print(f"Error adding {col_name}: {e}")
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    fix_schema()
