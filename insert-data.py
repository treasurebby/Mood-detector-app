import sqlite3

DB_PATH = "mood_data.db"
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Add your images and their mood labels
data = [
    ("dataset/happy/img1.jpg", "happy"),
    ("dataset/sad/img1.jpg", "sad"),
    ("dataset/angry/img1.jpg", "angry"),
    # Add all your images here
]

cursor.executemany("INSERT INTO moods (image_path, mood_label) VALUES (?, ?)", data)
conn.commit()
conn.close()
print("All image paths and labels inserted.")
