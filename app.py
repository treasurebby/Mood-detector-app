from flask import Flask, render_template, request, jsonify
import importlib
import sqlite3
import os
from datetime import datetime
from werkzeug.utils import secure_filename
import json
import numpy as np
import cv2

# Lazy-loaded keras model
_keras_model = None
_keras_labels = None

def find_model_path():
    """Return a candidate model path if present."""
    candidates = [
        os.path.join(os.getcwd(), 'emotion_detector_model.h5'),
        os.path.join(os.getcwd(), 'saved_models', 'mood_detector.h5'),
        os.path.join(os.getcwd(), 'saved_models', 'emotion_detector_model.h5')
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def load_labels_for_model(model_path):
    """Try to load labels from a JSON or txt file next to the model file."""
    base = os.path.splitext(model_path)[0]
    json_path = base + '_labels.json'
    txt_path = base + '_labels.txt'
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    if os.path.exists(txt_path):
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        except Exception:
            return None
    return None

def get_keras_model():
    """Lazy-load a Keras model if present on disk. Returns model or None."""
    global _keras_model, _keras_labels
    if _keras_model is not None:
        return _keras_model
    try:
        from tensorflow import keras
    except Exception as e:
        print(f"TensorFlow/Keras not available: {e}")
        return None

    model_path = find_model_path()
    if not model_path:
        return None

    try:
        _keras_model = keras.models.load_model(model_path)
        _keras_labels = load_labels_for_model(model_path)
        print(f"Keras model loaded from {model_path}")
        return _keras_model
    except Exception as e:
        print(f"Failed loading Keras model {model_path}: {e}")
        _keras_model = None
        return None

# Lazy-loaded heavy libraries
_deepface = None

def get_deepface():
    """Attempt to import DeepFace on demand. Return the module or None if unavailable."""
    global _deepface
    if _deepface is not None:
        return _deepface
    try:
        _deepface = importlib.import_module('deepface')
        return _deepface
    except Exception as e:
        print(f"DeepFace not available: {e}")
        _deepface = None
        return None

app = Flask(__name__)

# Folder for uploaded images
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --- DATABASE SETUP ---
DB_FILE = "emotions.db"

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            emotion TEXT,
            image_path TEXT,
            timestamp TEXT
        )
        """)
    print("✅ Database initialized")

init_db()

# --- ROUTES ---

@app.route("/")
def index():
    return render_template("index.html")


@app.route('/model_status')
def model_status():
    """Return status about whether the trained model file exists and whether DeepFace is available."""
    model_file = os.path.join(os.getcwd(), 'emotion_detector_model.h5')
    df = get_deepface()
    return jsonify({
        'deepface_available': df is not None,
        'model_file_exists': os.path.exists(model_file)
    })

@app.route("/detect", methods=["POST"])
def detect_emotion():
    try:
        name = request.form.get("name")
        image = request.files["image"]

        if not name or not image:
            return jsonify({"error": "Please provide both name and image"}), 400

        # Save image
        filename = secure_filename(image.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        image.save(filepath)

        # First try DeepFace if available
        df = get_deepface()
        emotion = 'Loading'
        try:
            if df is not None:
                analysis = df.DeepFace.analyze(img_path=filepath, actions=["emotion"], enforce_detection=False)
                if isinstance(analysis, dict):
                    emotion = analysis.get('dominant_emotion', 'Unknown')
                elif isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    emotion = analysis[0].get('dominant_emotion', 'Unknown')
                else:
                    emotion = 'Unknown'
        except Exception as e:
            print(f"DeepFace analysis failed: {e}")

        # If DeepFace not available or returned 'Loading', try Keras model if present
        if (df is None) or (emotion in ('Loading', 'Unknown')):
            km = get_keras_model()
            if km is not None:
                try:
                    # Preprocess image to model input
                    input_shape = km.input_shape
                    # input_shape may be (None, H, W, C) or (None, C, H, W)
                    if len(input_shape) == 4:
                        _, h, w, c = input_shape
                    elif len(input_shape) == 3:
                        # sometimes models omit batch dimension
                        h, w, c = input_shape
                    else:
                        # fallback
                        h, w, c = 64, 64, 3

                    img = cv2.imread(filepath)
                    if img is None:
                        raise ValueError('Failed to read saved image')
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (w, h))
                    img = img.astype('float32') / 255.0
                    # handle grayscale expected models
                    if c == 1:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        img = np.expand_dims(img, axis=-1)

                    x = np.expand_dims(img, axis=0)
                    preds = km.predict(x)
                    if preds.ndim == 2 and preds.shape[1] > 1:
                        idx = int(np.argmax(preds[0]))
                    else:
                        # regression/single-output
                        idx = int(np.argmax(preds))

                    # Map to label if available
                    if _keras_labels:
                        emotion = _keras_labels[idx]
                    else:
                        emotion = str(idx)
                except Exception as e:
                    print(f"Keras model prediction failed: {e}")

        # Save to DB
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute(
                "INSERT INTO detections (name, emotion, image_path, timestamp) VALUES (?, ?, ?, ?)",
                (name, emotion, filepath, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )

        # Map simple emoji for nicer UI
        emoji_map = {
            'angry': '😠', 'disgust': '🤢', 'fear': '😨', 'happy': '😊',
            'sad': '😢', 'surprise': '😲', 'neutral': '😐', 'Loading': '⏳',
            'Unknown': '❓', 'Model load error': '⚠️'
        }
        emoji = emoji_map.get(emotion.lower() if isinstance(emotion, str) else emotion, '🙂')

        # If the client prefers HTML, render a nice popup template; otherwise return JSON
        accept = request.headers.get('Accept', '')
        if 'text/html' in accept or 'application/xhtml+xml' in accept:
            return render_template('result.html', emotion=emotion, emoji=emoji)

        return jsonify({"name": name, "emotion": emotion, "emoji": emoji})

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict_alias():
    """Compatibility alias: some clients post to /predict — forward to /detect handler."""
    return detect_emotion()


if __name__ == "__main__":
    app.run(debug=True)
