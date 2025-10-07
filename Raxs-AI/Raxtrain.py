print("Content Moderation Ai System")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import Precision, Recall
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import cv2

class ContentModerationModel:
    def __init__(self, img_height=224, img_width=224):
        self.img_height = img_height
        self.img_width = img_width
        self.model = None

    def create_model(self):
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        base_model.trainable = False

        self.model = tf.keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])

        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', Precision(), Recall()]
        )

        return self.model

    def load_data(self, data_dir):
        images = []
        labels = []

        nude_path = os.path.join(data_dir, 'Nude Content')
        safe_path = os.path.join(data_dir, 'Safe Content')

        # Check if directories exist
        if not os.path.exists(nude_path):
            raise ValueError(f"Nude directory not found: {nude_path}")
        if not os.path.exists(safe_path):
            raise ValueError(f"Safe directory not found: {safe_path}")

        print("📥 Loading dataset...")
        
        # Load nude images
        for img_file in os.listdir(nude_path):
            img_path = os.path.join(nude_path, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (self.img_width, self.img_height))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(1)

        # Load safe images
        for img_file in os.listdir(safe_path):
            img_path = os.path.join(safe_path, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (self.img_width, self.img_height))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(0)

        print(f"✅ Loaded {len(images)} images ({sum(labels)} nude, {len(labels)-sum(labels)} safe)")
        return np.array(images), np.array(labels)

    def train(self, data_dir, epochs=20, batch_size=32):
        X, y = self.load_data(data_dir)
        X = X.astype('float32') / 255.0

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"📊 Training set: {X_train.shape[0]} images")
        print(f"📊 Validation set: {X_val.shape[0]} images")

        self.create_model()

        # Data Augmentation
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            zoom_range=0.15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
        )

        os.makedirs('../models', exist_ok=True)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=5, 
                restore_best_weights=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.2, 
                patience=3,
                monitor='val_loss'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                '../models/best_model.h5',
                save_best_only=True,
                monitor='val_accuracy',
                verbose=1
            )
        ]

        print("🎯 Starting training...")
        steps_per_epoch = max(len(X_train) // batch_size, 1)
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=steps_per_epoch,  
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        self.model.save('../models/final_model.keras')

        print("💾 Model saved successfully!")
        
        self.plot_training_history(history)
        return history

    def plot_training_history(self, history):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(history.history['loss'], label='Training Loss')
        axes[1].plot(history.history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig('../models/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

    def predict_image(self, img_path):
        """Predict single image - returns detailed results"""
        img = cv2.imread(img_path)
        if img is None:
            return {"error": "Could not load image"}

        img = cv2.resize(img, (self.img_width, self.img_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0) / 255.0

        pred = self.model.predict(img, verbose=0)[0][0]
        
        result = {
            "prediction": float(pred),
            "class": "Nude" if pred > 0.5 else "Safe",
            "confidence": float(pred) if pred > 0.5 else float(1 - pred),
            "is_nude": bool(pred > 0.5)
        }
        
        print(f"🎯 Prediction: {result['class']} (confidence: {result['confidence']:.2%})")
        return result

    def predict_video(self, video_path, frame_interval=10, threshold=0.2):
        """Predict video - returns detailed analysis"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Could not open video file"}

        frame_count = 0
        nude_frames = 0
        total_frames = 0
        frame_predictions = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_interval != 0:
                continue

            total_frames += 1
            frame_resized = cv2.resize(frame, (self.img_width, self.img_height))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_input = np.expand_dims(frame_rgb, axis=0) / 255.0

            pred = self.model.predict(frame_input, verbose=0)[0][0]
            is_nude = pred > 0.5
            
            if is_nude:
                nude_frames += 1
                
            frame_predictions.append({
                "frame": frame_count,
                "prediction": float(pred),
                "is_nude": bool(is_nude)
            })

        cap.release()

        nude_ratio = nude_frames / max(total_frames, 1)
        is_video_nude = nude_ratio > threshold
        
        result = {
            "class": "Nude" if is_video_nude else "Safe",
            "nude_ratio": float(nude_ratio),
            "nude_frames": nude_frames,
            "total_frames_analyzed": total_frames,
            "is_nude": bool(is_video_nude),
            "frame_predictions": frame_predictions
        }

        print(f"🎯 Video Analysis: {result['class']}")
        print(f"📊 Nude frames: {nude_frames}/{total_frames} ({nude_ratio:.2%})")
        return result


if __name__ == "__main__":
    model = ContentModerationModel()
    history = model.train(r"D:\Content_dataset", epochs=20)