"""
ML Models: Classification và EQ Suggestion
Sử dụng YAMNet để extract embedding, sau đó predict bằng custom models.
"""

import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import soundfile as sf
import librosa
from typing import Tuple, Optional, List

# YAMNet yêu cầu sample rate 16kHz
YAMNET_SAMPLE_RATE = 16000

# EQ bands (9 bands)
EQ_BANDS = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]

# Classification labels (có thể load từ file nếu có)
DEFAULT_LABELS = [
    "Music", "Vocal", "Podcast", "EDM", "Rock", 
    "Classical", "Jazz", "Hip-Hop", "Country", "Blues"
]


class MLModelManager:
    """Quản lý việc load và sử dụng ML models."""
    
    def __init__(self, models_dir: str = "models"):
        # Chuyển relative path thành absolute path nếu cần
        if not os.path.isabs(models_dir):
            # Nếu là relative path, tính từ project root
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, models_dir)
        self.models_dir = models_dir
        self.yamnet = None
        self.classification_model = None
        self.eq_suggestion_model = None
        self.labels = None
        self._initialized = False
    
    def initialize(self):
        """Khởi tạo models (lazy loading)."""
        if self._initialized:
            return
        
        try:
            # Load YAMNet
            print("Loading YAMNet...")
            self.yamnet = hub.load(
                "https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1"
            )
            print("YAMNet loaded successfully")
            
            # Load Classification model
            classification_path = os.path.join(self.models_dir, "classification.keras")
            if os.path.exists(classification_path):
                print(f"Loading classification model from {classification_path}...")
                self.classification_model = tf.keras.models.load_model(classification_path)
                print("Classification model loaded successfully")
            else:
                print(f"Warning: Classification model not found at {classification_path}")
            
            # Load EQ Suggestion model
            eq_suggestion_path = os.path.join(self.models_dir, "EQSuggestion.keras")
            if os.path.exists(eq_suggestion_path):
                print(f"Loading EQ suggestion model from {eq_suggestion_path}...")
                self.eq_suggestion_model = tf.keras.models.load_model(eq_suggestion_path)
                print("EQ suggestion model loaded successfully")
            else:
                print(f"Warning: EQ suggestion model not found at {eq_suggestion_path}")
            
            # Load labels
            labels_path = os.path.join(self.models_dir, "label_classes.npy")
            if os.path.exists(labels_path):
                self.labels = np.load(labels_path, allow_pickle=True).tolist()
                print(f"Loaded {len(self.labels)} labels from file")
            else:
                # Sử dụng default labels hoặc infer từ model
                if self.classification_model:
                    # Nếu model có output shape, dùng số lượng đó
                    output_shape = self.classification_model.output_shape
                    if output_shape and len(output_shape) > 0:
                        n_classes = output_shape[-1]
                        if n_classes <= len(DEFAULT_LABELS):
                            self.labels = DEFAULT_LABELS[:n_classes]
                        else:
                            self.labels = DEFAULT_LABELS + [f"Class_{i}" for i in range(len(DEFAULT_LABELS), n_classes)]
                    else:
                        self.labels = DEFAULT_LABELS
                else:
                    self.labels = DEFAULT_LABELS
                print(f"Using default labels: {self.labels}")
            
            self._initialized = True
        except Exception as e:
            print(f"Error initializing models: {e}")
            raise
    
    def load_audio_for_yamnet(self, path: str) -> np.ndarray:
        """
        Load audio và chuẩn hóa cho YAMNet (16kHz, mono, float32).
        
        Args:
            path: Đường dẫn file audio
            
        Returns:
            Audio array (float32, 16kHz, mono)
        """
        audio, sr = sf.read(path, always_2d=False)
        
        # Chuyển stereo → mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        # Resample về 16kHz nếu cần
        if sr != YAMNET_SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=YAMNET_SAMPLE_RATE)
        
        return audio.astype(np.float32)
    
    def extract_embedding(self, wav: np.ndarray) -> tf.Tensor:
        """
        Extract embedding từ YAMNet.
        
        Args:
            wav: Audio array (16kHz, mono, float32)
            
        Returns:
            Embedding tensor shape (1, 1024)
        """
        if self.yamnet is None:
            raise RuntimeError("YAMNet not initialized. Call initialize() first.")
        
        # YAMNet trả về (scores, embeddings, spectrogram)
        _, emb, _ = self.yamnet(wav)
        
        # Average pooling: (n_frames, 1024) → (1, 1024)
        return tf.reduce_mean(emb, axis=0)[None, :]
    
    def classify_audio(self, audio_path: str) -> Tuple[str, float, List[float]]:
        """
        Phân loại audio thành label.
        
        Args:
            audio_path: Đường dẫn file audio
            
        Returns:
            (predicted_label, confidence, all_probabilities)
        """
        if self.classification_model is None:
            raise RuntimeError("Classification model not loaded")
        
        if not self._initialized:
            self.initialize()
        
        # Load audio và extract embedding
        wav = self.load_audio_for_yamnet(audio_path)
        X = self.extract_embedding(wav)
        
        # Predict
        probs = self.classification_model.predict(X, verbose=0)[0]
        idx = np.argmax(probs)
        
        predicted_label = self.labels[idx] if idx < len(self.labels) else f"Class_{idx}"
        confidence = float(probs[idx])
        
        return predicted_label, confidence, probs.tolist()
    
    def suggest_eq(self, audio_path: str) -> List[float]:
        """
        Đề xuất EQ preset từ audio.
        
        Args:
            audio_path: Đường dẫn file audio
            
        Returns:
            List 9 giá trị EQ gains (dB) cho 9 bands
        """
        if self.eq_suggestion_model is None:
            raise RuntimeError("EQ suggestion model not loaded")
        
        if not self._initialized:
            self.initialize()
        
        # Load audio và extract embedding
        wav = self.load_audio_for_yamnet(audio_path)
        X = self.extract_embedding(wav)
        
        # Predict EQ (output: normalized [0, 1] cho 9 bands)
        eq_norm = self.eq_suggestion_model.predict(X, verbose=0)[0]
        
        # Denormalize: [0, 1] → [-12, +12] dB
        eq_db = (eq_norm * 24) - 12  # 0 → -12dB, 1 → +12dB
        
        # Round về 1 chữ số thập phân
        eq_db = [round(float(g), 1) for g in eq_db]
        
        return eq_db


# Global instance (lazy initialization)
_model_manager: Optional[MLModelManager] = None


def get_model_manager(models_dir: str = "models") -> MLModelManager:
    """Get hoặc tạo global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = MLModelManager(models_dir=models_dir)
    return _model_manager

