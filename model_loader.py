"""
model_loader.py
------------------------------------
Load semua model AI dan label.
Menggunakan pola Singleton Python murni (tanpa st.cache_resource) 
agar aman dipanggil dari dalam thread WebRTC.
"""

import json
import tensorflow as tf

from config import (
    CNN_MODEL_PATH,
    LSTM_MODEL_PATH,
    CNN_LABEL_PATH,
    LSTM_LABEL_PATH,
    TF_NUM_THREADS,
)

try:
    tf.config.threading.set_intra_op_parallelism_threads(TF_NUM_THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(TF_NUM_THREADS)
except RuntimeError:
    pass

class ModelManager:
    """
    Menyimpan semua model AI beserta labelnya.
    """
    def __init__(self):
        self.cnn_model = None
        self.lstm_model = None
        self.cnn_labels = {}
        self.lstm_labels = {}

    def load(self):
        if self.cnn_model is None:
            self.cnn_model = tf.keras.models.load_model(
                CNN_MODEL_PATH,
                compile=False
            )

        if self.lstm_model is None:
            self.lstm_model = tf.keras.models.load_model(
                LSTM_MODEL_PATH,
                compile=False
            )

        if not self.cnn_labels:
            with open(CNN_LABEL_PATH, "r", encoding="utf-8") as f:
                self.cnn_labels = json.load(f)

        if not self.lstm_labels:
            with open(LSTM_LABEL_PATH, "r", encoding="utf-8") as f:
                self.lstm_labels = json.load(f)

        return self

    def get_cnn(self):
        return self.cnn_model

    def get_lstm(self):
        return self.lstm_model

    def get_cnn_labels(self):
        return self.cnn_labels

    def get_lstm_labels(self):
        return self.lstm_labels

# ==========================================
# SINGLETON PATTERN PENGGANTI st.cache
# ==========================================
_model_manager_instance = None

def load_models():
    """Mengembalikan instance ModelManager yang sama setiap kali dipanggil."""
    global _model_manager_instance
    if _model_manager_instance is None:
        _model_manager_instance = ModelManager().load()
    return _model_manager_instance
