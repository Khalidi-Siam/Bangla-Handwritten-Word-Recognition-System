import os
import json
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# =========================================================
# CONFIG
# =========================================================

DATASET_PATH = Path("BanglaLekha-Isolated") / "Images"  # Update this path to your dataset location

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 1

LEARNING_RATE = 1e-4

MODEL_SAVE_PATH = "model.keras"
LABELS_SAVE_PATH = "labels.json"

SEED = 42

# =========================================================
# FIX RANDOMNESS
# =========================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================================================
# LOAD DATASET
# =========================================================

"""
Expected dataset structure:

BanglaLekha-Isolated/
    character_1/
        image1.png
        image2.png
    character_2/
        image1.png
        ...

Each folder name will be used as label.
"""

image_paths = []
labels = []

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")

for class_name in sorted(os.listdir(DATASET_PATH)):

    class_dir = os.path.join(DATASET_PATH, class_name)

    if not os.path.isdir(class_dir):
        continue

    for file_name in os.listdir(class_dir):

        if file_name.lower().endswith(SUPPORTED_EXTENSIONS):

            file_path = os.path.join(class_dir, file_name)

            image_paths.append(file_path)
            labels.append(class_name)

print(f"Total Images: {len(image_paths)}")
print(f"Total Classes: {len(set(labels))}")

# =========================================================
# LABEL ENCODING
# =========================================================

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

num_classes = len(label_encoder.classes_)

print(f"Encoded Classes: {num_classes}")

# Save label mapping
label_mapping = {
    str(i): label
    for i, label in enumerate(label_encoder.classes_)
}

with open(LABELS_SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(label_mapping, f, ensure_ascii=False, indent=4)

print(f"Labels saved to: {LABELS_SAVE_PATH}")

# =========================================================
# TRAIN / VALIDATION SPLIT
# =========================================================

train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths,
    encoded_labels,
    test_size=0.2,
    random_state=SEED,
    stratify=encoded_labels
)

print(f"Train Samples: {len(train_paths)}")
print(f"Validation Samples: {len(val_paths)}")

# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image_path, label):

    image = tf.io.read_file(image_path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

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

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_paths, train_labels)
)

train_dataset = (
    train_dataset
    .shuffle(1000)
    .map(preprocess_image, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (val_paths, val_labels)
)

val_dataset = (
    val_dataset
    .map(preprocess_image, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

# =========================================================
# BUILD MODEL (TRANSFER LEARNING)
# =========================================================

base_model = DenseNet121(
    include_top=False,
    weights="imagenet",
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)

base_model.trainable = False

inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

x = data_augmentation(inputs)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

x = layers.Dense(256, activation="relu")(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)

# =========================================================
# COMPILE MODEL
# =========================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================================================
# CALLBACKS
# =========================================================

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1
    ),

    ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
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

print(f"\nModel saved to: {MODEL_SAVE_PATH}")
print("Training completed successfully.")