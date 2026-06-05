import numpy as np
import tensorflow as tf
import json
from PIL import Image
from config import settings

class BengaliWordPredictor:
    def __init__(self):
        self.model_path = settings.predict.model_path
        self.labels_json_path = settings.predict.labels_json_path
        self.image_size = settings.predict.image_size
        self.model = None
        self.index_to_label = {}
        
        self.load_model_and_labels()
        
    def load_model_and_labels(self):
        self.model = tf.keras.models.load_model(self.model_path)
        
        with open(self.labels_json_path, "r", encoding="utf-8") as f:
            label_mapping = json.load(f)

        # Inner labels.json (from training) is already 0-indexed exactly as the model predicts
        self.index_to_label = {int(k): v for k, v in label_mapping.items()}
        
    def preprocess_image(self, image: Image.Image):
        image = image.convert("L")
        image = image.resize((self.image_size, self.image_size))
        image = np.array(image).astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=-1)
        return image
        
    def predict(self, image: Image.Image):
        input_tensor = self.preprocess_image(image)
        
        predictions = self.model.predict(input_tensor, verbose=0)
        predicted_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))
        
        predicted_label = self.index_to_label[predicted_index]
        
        return predicted_label, confidence
