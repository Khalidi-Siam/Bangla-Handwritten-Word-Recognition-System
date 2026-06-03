import os
import json
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from pathlib import Path
from sklearn.model_selection import train_test_split

# =========================================================
# CONFIG
# =========================================================

DATASET_PATH = Path("BanglaLekha-Isolated") / "Images"  
LABELS_JSON_PATH = Path("labels.json") 
IMAGE_SIZE = 224  
BATCH_SIZE = 32
EPOCHS = 1  
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = "models/model.keras"
SEED = 42

# Optional: use subset of images per class for fast experiments
SAMPLE_PER_CLASS = 100  # e.g., 500, or None for all images

# =========================================================
# FIX RANDOMNESS
# =========================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================================================
# LOAD MANUAL LABEL MAPPING
# =========================================================

with open(LABELS_JSON_PATH, "r", encoding="utf-8") as f:
    label_mapping = json.load(f) 

# Create deterministic mapping (0-indexed for training)
# Class keys from folder names ("1", "2", ...) mapped to 0, 1, ...
sorted_class_ids = sorted(label_mapping.keys(), key=int)
class_id_to_index = {class_id: idx for idx, class_id in enumerate(sorted_class_ids)}
index_to_label = {idx: label_mapping[class_id] for class_id, idx in class_id_to_index.items()}

print(f"Loaded {len(index_to_label)} labels from {LABELS_JSON_PATH}")

# =========================================================
# LOAD DATASET
# =========================================================

image_paths = []
encoded_labels = []

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")

for class_name in sorted(os.listdir(DATASET_PATH)):
    class_dir = os.path.join(DATASET_PATH, class_name)
    if not os.path.isdir(class_dir):
        continue

    # Skip any label not in manual mapping
    if class_name not in class_id_to_index:
        continue

    all_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.lower().endswith(SUPPORTED_EXTENSIONS)]

    # Sample subset if specified
    if SAMPLE_PER_CLASS is not None:
        all_files = random.sample(all_files, min(SAMPLE_PER_CLASS, len(all_files)))

    image_paths.extend(all_files)
    encoded_labels.extend([class_id_to_index[class_name]] * len(all_files))

print(f"Total images: {len(image_paths)}")
print(f"Total classes used: {len(set(encoded_labels))}")

# =========================================================
# TRAIN / VALIDATION SPLIT
# =========================================================

train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, encoded_labels, test_size=0.2, random_state=SEED, stratify=encoded_labels
)

print(f"Train samples: {len(train_paths)}")
print(f"Validation samples: {len(val_paths)}")

# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# =========================================================
# DATA AUGMENTATION
# =========================================================

data_augmentation = models.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.05, 0.05),
    layers.RandomContrast(0.1)
])

# =========================================================
# TF DATASET
# =========================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_dataset = train_dataset.shuffle(1000).map(preprocess_image, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_dataset = val_dataset.map(preprocess_image, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

# =========================================================
# BUILD MODEL
# =========================================================

num_classes = len(index_to_label)

base_model = DenseNet121(include_top=False, weights="imagenet", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
base_model.trainable = False  # Freeze base

inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

# =========================================================
# COMPILE MODEL
# =========================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================================================
# CALLBACKS
# =========================================================

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    ModelCheckpoint(filepath=MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True, verbose=1)
]

# =========================================================
# TRAIN
# =========================================================

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =========================================================
# SAVE FINAL MODEL
# =========================================================

model.save(MODEL_SAVE_PATH)
print(f"\n✅ Model saved to: {MODEL_SAVE_PATH}")
print("🎉 Training completed successfully!")