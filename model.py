import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# --- Parameters ---
IMG_SIZE = (64, 64)  # Resize images
BATCH_SIZE = 32
EPOCHS = 10

# --- Helper Functions ---
def load_images_from_folder(folder):
    image_paths = []
    labels = []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if not os.path.isdir(label_folder):
            continue
        for filename in os.listdir(label_folder):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                image_paths.append(os.path.join(label_folder, filename))
                labels.append(label)
    return image_paths, labels

def preprocess_images(image_paths):
    images = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        img = img.resize(IMG_SIZE)
        images.append(np.array(img))
    return np.array(images) / 255.0  # Normalize to [0,1]

def encode_labels(labels):
    # Use a deterministic ordering for labels so mapping is reproducible
    unique_labels = sorted(list(set(labels)))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    indices = [label_to_index[label] for label in labels]
    return to_categorical(indices), unique_labels

# --- Load Train & Test Data ---
train_folder = "dataset/train"
test_folder = "dataset/test"

train_paths, train_labels = load_images_from_folder(train_folder)
test_paths, test_labels = load_images_from_folder(test_folder)

X_train = preprocess_images(train_paths)
y_train, label_names = encode_labels(train_labels)

X_test = preprocess_images(test_paths)
y_test, _ = encode_labels(test_labels)

# --- Build Model ---
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(label_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- Train Model ---
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)

# --- Save Model ---
model.save('mood_detector_model.h5')
print("Model trained and saved successfully!")
# Save label names next to the model so the web app can map predictions to emotions
try:
    import json
    base = os.path.splitext('mood_detector_model.h5')[0]
    labels_path = base + '_labels.json'
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(label_names, f, ensure_ascii=False)
    print(f"Saved label file to {labels_path}")
except Exception as e:
    print(f"Failed to save labels file: {e}")
