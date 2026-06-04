import os
import json
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from config import settings
from sklearn.model_selection import train_test_split


class BengaliWordTrainer:
    def __init__(self):
        self.dataset_path = settings.data.dataset_path
        self.labels_json_path = settings.data.labels_json_path
        self.image_size = settings.train.image_size
        self.batch_size = settings.train.batch_size
        self.epochs = settings.train.epochs
        self.learning_rate = settings.train.learning_rate
        self.model_save_path = settings.train.model_save_path
        self.seed = settings.seed
        self.sample_per_class = settings.train.sample_per_class
        
        self.class_id_to_index = {}
        self.index_to_label = {}
        self.model = None

        self._fix_randomness()

    def _fix_randomness(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def load_label_mapping(self):
        with open(self.labels_json_path, "r", encoding="utf-8") as f:
            label_mapping = json.load(f) 

        # The outer labels.json is 0-indexed.
        # But our datasets folders are 1-indexed (e.g. folder "1", "2").
        # We process label_mapping mapping integer keys.
        self.index_to_label = {int(k): v for k, v in label_mapping.items()}
        
        # Determine class directory to index mapping (e.g., folder "1" maps to index 0)
        self.class_dir_to_index = {}
        for idx in sorted(self.index_to_label.keys()):
            # the folder names are index + 1
            folder_name = str(idx + 1)
            self.class_dir_to_index[folder_name] = idx

        print(f"Loaded {len(self.index_to_label)} labels from {self.labels_json_path}")

    def prepare_datasets(self):
        image_paths = []
        encoded_labels = []
        supported_extensions = (".png", ".jpg", ".jpeg", ".bmp")

        for class_name in sorted(os.listdir(self.dataset_path)):
            class_dir = os.path.join(self.dataset_path, class_name)
            if not os.path.isdir(class_dir):
                continue

            # Skip any label not in manual mapping
            if class_name not in self.class_dir_to_index:
                continue

            all_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.lower().endswith(supported_extensions)]

            # Sample subset if specified
            if self.sample_per_class is not None:
                all_files = random.sample(all_files, min(self.sample_per_class, len(all_files)))

            image_paths.extend(all_files)
            encoded_labels.extend([self.class_dir_to_index[class_name]] * len(all_files))

        print(f"Total images: {len(image_paths)}")
        print(f"Total classes used: {len(set(encoded_labels))}")

        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, encoded_labels, test_size=0.2, random_state=self.seed, stratify=encoded_labels
        )

        print(f"Train samples: {len(train_paths)}")
        print(f"Validation samples: {len(val_paths)}")

        autotune = tf.data.AUTOTUNE

        train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
        train_dataset = train_dataset.shuffle(1000).map(self._preprocess_image, num_parallel_calls=autotune).batch(self.batch_size).prefetch(autotune)

        val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
        val_dataset = val_dataset.map(self._preprocess_image, num_parallel_calls=autotune).batch(self.batch_size).prefetch(autotune)

        return train_dataset, val_dataset

    def _preprocess_image(self, image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, (self.image_size, self.image_size))
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    def build_model(self):
        num_classes = len(self.index_to_label)

        data_augmentation = models.Sequential([
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.05, 0.05),
            layers.RandomContrast(0.1)
        ])

        base_model = DenseNet121(include_top=False, weights="imagenet", input_shape=(self.image_size, self.image_size, 3))
        base_model.trainable = False  # Freeze base

        inputs = layers.Input(shape=(self.image_size, self.image_size, 3))
        x = data_augmentation(inputs)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)

        self.model = models.Model(inputs, outputs)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        self.model.summary()

    def train(self, train_dataset, val_dataset):
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
            ModelCheckpoint(filepath=self.model_save_path, monitor="val_accuracy", save_best_only=True, verbose=1)
        ]

        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.epochs,
            callbacks=callbacks
        )
        return history

    def save_model(self):
        self.model.save(self.model_save_path)
        
        # Save a model-specific labels.json mapping in the same directory as the model
        model_dir = os.path.dirname(self.model_save_path)
        if not model_dir:
            model_dir = "."
        model_labels_path = os.path.join(model_dir, "labels.json")
        with open(model_labels_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self.index_to_label.items()}, f, ensure_ascii=False, indent=4)
            
        print(f"\n✅ Model saved to: {self.model_save_path}")
        print(f"✅ Prediction labels mapping saved to: {model_labels_path}")
        print("🎉 Training completed successfully!")
        
    def run(self):
        self.load_label_mapping()
        train_dataset, val_dataset = self.prepare_datasets()
        self.build_model()
        self.train(train_dataset, val_dataset)
        self.save_model()

if __name__ == "__main__":
    trainer = BengaliWordTrainer()
    print("\n\n*** Summary of training configuration ***")
    print(f"Model Save Path: {trainer.model_save_path}")
    print(f"Image Size: {trainer.image_size}")
    print(f"Batch Size: {trainer.batch_size}")
    print(f"Learning Rate: {trainer.learning_rate}")
    print(f"Epochs: {trainer.epochs}\n\n")
    print("🚀 Starting training...")
    trainer.run()