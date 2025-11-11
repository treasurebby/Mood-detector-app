from keras.models import load_model
import json
import numpy as np
from tensorflow.keras.preprocessing import image
import sys
import os

# ----- CONFIG -----
MODEL_PATH = "mood_detector_model.h5"       # or .keras if you saved in new format
LABELS_PATH = "mood_detector_model_labels.json"
IMAGE_PATH = ""  # we’ll set this from command line

# ----- LOAD MODEL & LABELS -----
if not os.path.exists(MODEL_PATH):
    print(f"Model file not found: {MODEL_PATH}")
    sys.exit(1)

model = load_model(MODEL_PATH)
with open(LABELS_PATH, "r") as f:
    labels = json.load(f)

# ----- LOAD IMAGE -----
if len(sys.argv) < 2:
    print("Usage: python predict.py <image_path>")
    sys.exit(1)

IMAGE_PATH = sys.argv[1]

if not os.path.exists(IMAGE_PATH):
    print(f"Image not found: {IMAGE_PATH}")
    sys.exit(1)

img = image.load_img(IMAGE_PATH, target_size=(224, 224))  # match model input
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize if needed

# ----- PREDICT -----
preds = model.predict(img_array)
predicted_index = int(np.argmax(preds))
predicted_label = labels[str(predicted_index)]  # match JSON keys

# ----- OPTIONAL: Map emoji -----
emoji_map = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😡",
    "surprised": "😲",
    "neutral": "😐"
}
emoji = emoji_map.get(predicted_label.lower(), "🤔")

# ----- OUTPUT -----
print(f"Predicted Emotion: {predicted_label} {emoji}")
