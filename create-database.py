import sqlite3

DB_PATH = "mood_data.db"  # same path as in model.py
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Create the table
cursor.execute("""
CREATE TABLE IF NOT EXISTS moods (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT NOT NULL,
    mood_label TEXT NOT NULL
)
""")

conn.commit()
conn.close()
print("Table 'moods' created (if it didn't exist).")
