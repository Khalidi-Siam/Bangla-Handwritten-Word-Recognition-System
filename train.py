import os
import csv
import json
import random
import numpy as np
import tensorflow as tf
import mlflow

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
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
        self.save_json_path = settings.train.save_json_path
        self.save_summary_path = settings.train.save_summary_path
        self.seed = settings.seed
        self.sample_per_class = settings.train.sample_per_class
        
        self.class_id_to_index = {}
        self.index_to_label = {}
        self.model = None
        self.data_augmentation = tf.keras.Sequential([
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.05, 0.05),
            layers.RandomContrast(0.1)
        ])
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

            all_files.sort()
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
        train_dataset = train_dataset.shuffle(1000).map(
            lambda x, y: self._preprocess_image(x, y, training=True),
            num_parallel_calls=autotune
        ).batch(self.batch_size).prefetch(autotune)

        val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
        val_dataset = val_dataset.map(
            lambda x, y: self._preprocess_image(x, y, training=False),
            num_parallel_calls=autotune
        ).batch(self.batch_size).prefetch(autotune)

        return train_dataset, val_dataset

    def _preprocess_image(self, image_path, label, training=False):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=1, expand_animations=False)
        image = tf.image.resize(image, (self.image_size, self.image_size))
        image = tf.cast(image, tf.float32) / 255.0

        if training:
            image = self.data_augmentation(image, training=True)

        return image, label

    
    def build_model(self):
        num_classes = len(self.index_to_label)

        inputs = layers.Input(shape=(self.image_size, self.image_size, 1))

        # ── Block 1 ──────────────────────────────────────────────────────────
        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(0.20)(x)

        # ── Block 2 ──────────────────────────────────────────────────────────
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(0.25)(x)

        # ── Block 3 ──────────────────────────────────────────────────────────
        x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(0.30)(x)

        # ── Block 4 ──────────────────────────────────────────────────────────
        x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(0.35)(x)

        # ── Head ─────────────────────────────────────────────────────────────
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.50)(x)
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
            ModelCheckpoint(filepath=self.model_save_path, monitor="val_loss", save_best_only=True, verbose=1)
        ]

        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.epochs,
            callbacks=callbacks
        )
        return history

    def save_model(self):
        # Create directory if it does not exist
        os.makedirs(os.path.dirname(self.save_json_path), exist_ok=True)
        # Save a model-specific labels.json mapping in the same directory as the model
        with open(self.save_json_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self.index_to_label.items()}, f, ensure_ascii=False, indent=4)
            
        print(f"\n✅ Model saved to: {self.model_save_path}")
        print(f"✅ Prediction labels mapping saved to: {self.save_json_path}")
        print("🎉 Training completed successfully!")

    def save_history_csv(self, history):
        """Write per-epoch train/val metrics to a CSV file."""
        os.makedirs(os.path.dirname(self.save_summary_path), exist_ok=True)

        fieldnames = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"]
        rows = zip(
            range(1, len(history.history["loss"]) + 1),
            history.history["loss"],
            history.history["accuracy"],
            history.history["val_loss"],
            history.history["val_accuracy"],
        )

        with open(self.save_summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for epoch, t_loss, t_acc, v_loss, v_acc in rows:
                writer.writerow({
                    "epoch": epoch,
                    "train_loss": round(float(t_loss), 6),
                    "train_accuracy": round(float(t_acc), 6),
                    "val_loss": round(float(v_loss), 6),
                    "val_accuracy": round(float(v_acc), 6),
                })

        print(f"✅ Training history CSV saved to: {self.save_summary_path}")
        
    def run(self):
        mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
        mlflow.set_registry_uri(settings.mlflow.registry_uri)
        mlflow.set_experiment(settings.mlflow.experiment_name)


        with mlflow.start_run():
            mlflow.log_params({
                "image_size": self.image_size,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "seed": self.seed,
                "sample_per_class": self.sample_per_class
            })
            
            
            self.load_label_mapping()
            train_dataset, val_dataset = self.prepare_datasets()
            self.build_model()

            history = self.train(train_dataset, val_dataset)

            self.save_model()
            self.save_history_csv(history)

            # 1. Find best epoch (based on validation accuracy)
            best_epoch = int(np.argmin(history.history["val_loss"]))

            # 2. Extract BEST metrics (val + train from same epoch)
            best_val_acc = history.history["val_accuracy"][best_epoch]
            best_val_loss = history.history["val_loss"][best_epoch]

            best_train_acc = history.history["accuracy"][best_epoch]
            best_train_loss = history.history["loss"][best_epoch]

            # 3. Log to MLflow
            mlflow.log_metrics({
                "best_epoch": best_epoch,
                "best_val_accuracy": best_val_acc,
                "best_val_loss": best_val_loss,
                "best_train_accuracy": best_train_acc,
                "best_train_loss": best_train_loss
            })

            # 4. Load BEST saved model (from ModelCheckpoint)
            best_model = tf.keras.models.load_model(self.model_save_path)

            # 5. Log model
            mlflow.keras.log_model(best_model, name="model")
            
            # Log the labels.json as an artifact to mlflow
            mlflow.log_artifact(self.save_json_path, artifact_path="model")

            # Log per-epoch CSV history as an artifact
            mlflow.log_artifact(self.save_summary_path, artifact_path="metrics")

# if __name__ == "__main__":
trainer = BengaliWordTrainer()
print("\n\n*** Summary of training configuration ***")
print(f"Model Save Path: {trainer.model_save_path}")
print(f"Image Size: {trainer.image_size}")
print(f"Batch Size: {trainer.batch_size}")
print(f"Learning Rate: {trainer.learning_rate}")
print(f"Epochs: {trainer.epochs}")
print(f"Tracking URI: {settings.mlflow.tracking_uri}")
print(f"per class sample limit: {trainer.sample_per_class}\n\n")
print("🚀 Starting training...")
trainer.run()