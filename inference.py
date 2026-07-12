"""
inference.py
---------------------------------------
Inference Engine
CNN + BiLSTM
"""

from collections import deque

import numpy as np

from config import (
    CNN_THRESHOLD,
    LSTM_THRESHOLD,
    SEQUENCE_LENGTH,
    CNN_INPUT_SIZE,
    LANDMARK_VECTOR_SIZE,
)

from model_loader import load_models


class InferenceEngine:

    def __init__(self):
        self.models = load_models()
        self.cnn = self.models.get_cnn()
        self.lstm = self.models.get_lstm()
        self.cnn_labels = self.models.get_cnn_labels()
        self.lstm_labels = self.models.get_lstm_labels()
        self.sequence = deque(maxlen=SEQUENCE_LENGTH)
        self.cnn_input = CNN_INPUT_SIZE
        self.lstm_input = LANDMARK_VECTOR_SIZE

    # =====================================================
    # CNN PREDICT INPUT: (126,) OUTPUT: LABEL + CONFIDENCE
    # =====================================================

    def predict_cnn(self, landmarks):
        """
        Predict huruf menggunakan landmark MediaPipe Hands (126 fitur)
        """
        landmarks = np.asarray(
            landmarks,
            dtype=np.float32
        )

        if landmarks.size != self.cnn_input:
            return "Unknown", 0.0

        landmarks = landmarks.reshape(1, self.cnn_input)

        try:
            prediction = self.cnn.predict(
                landmarks,
                verbose=0
            )[0]
        except Exception:
            return "Unknown", 0.0
        
        idx = np.argmax(prediction)
        confidence = float(prediction[idx])
        label = self.cnn_labels.get(str(int(idx)), "Unknown")

        if confidence < CNN_THRESHOLD:
            label = "Unknown"

        return label, confidence

    # =====================================================
    # LSTM INPUT: (30, 258) OUTPUT: LABEL + CONFIDENCE
    # =====================================================

    def add_landmarks(self, landmarks):

        landmarks = np.asarray(
            landmarks,
            dtype=np.float32
        )

        if landmarks.size != self.lstm_input:
            return

        self.sequence.append(
            landmarks
        )

    def is_sequence_ready(self):
        return len(self.sequence) == SEQUENCE_LENGTH

    def predict_lstm(self):
        if not self.is_sequence_ready():
            return None, 0.0

        sequence = np.asarray(
            self.sequence,
            dtype=np.float32
        )

        if sequence.shape != (SEQUENCE_LENGTH, self.lstm_input):
            return "Unknown", 0.0

        data = np.expand_dims(
            sequence,
            axis=0
        )

        try:
            prediction = self.lstm.predict(
                data,
                verbose=0
            )[0]
        except Exception:
            return "Unknown", 0.0

        idx = int(np.argmax(prediction))
        confidence = float(prediction[idx])
        label = self.lstm_labels.get(str(idx), "Unknown")

        if confidence < LSTM_THRESHOLD:
            label = "Unknown"

        return label, confidence
    
    def reset(self):
        self.sequence.clear()