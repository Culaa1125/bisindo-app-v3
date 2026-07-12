import av
import cv2
import time
import threading

import numpy as np

from collections import deque
from collections import Counter

from streamlit_webrtc import VideoProcessorBase

from detectors import (
    HolisticManager,
    extract_landmarks,
    extract_cnn_landmarks,
    draw_landmarks,
    has_hand,
)

from inference import InferenceEngine

from config import FRAME_SKIP, PROCESS_WIDTH

class BISINDOProcessor(VideoProcessorBase):

    def __init__(self):
        self._lock = threading.Lock()
        self.detector = HolisticManager()
        self.inference = InferenceEngine()
        self.previous_landmark = None
        self.motion_score = 0.0
        self.last_prediction = ""
        self.result_mode = ""
        self.confidence = 0.0
        self.last_time = time.time()
        self.fps = 0

        # Frame-skip: hanya jalankan Holistic + model setiap
        # FRAME_SKIP frame. Frame yang di-skip memakai hasil
        # deteksi (landmark) terakhir untuk digambar, jadi video
        # tetap terlihat mengalir walau inferensi lebih jarang.
        self._frame_index = 0
        self._last_results = None

        # Setting dari UI
        self.conf_cnn = 0.80
        self.conf_lstm = 0.65
        self.motion_low = 0.005
        self.motion_high = 0.015
        self.blur_bg = False

        # STATE MACHINE
        self.state = "IDLE"

        # Motion Counter
        self.motion_counter = 0
        self.still_counter = 0

        # jumlah frame sebelum pindah mode
        self.motion_trigger = 8

        # jumlah frame diam sebelum kembali ke CNN
        self.still_trigger = 25
        self.motion_buffer = deque(maxlen=8)

        # Prediction Buffer
        self.cnn_buffer = deque(maxlen=5)
        self.lstm_buffer = deque(maxlen=5)
        self.stable_prediction = ""
        self.stable_confidence = 0.0
        self.stable_mode = ""
    
    def update_settings(
        self,
        conf_cnn,
        conf_lstm,
        motion_low,
        motion_high,
    ):
        self.conf_cnn = conf_cnn
        self.conf_lstm = conf_lstm
        self.motion_low = motion_low
        self.motion_high = motion_high

    def calculate_fps(self):
        current = time.time()
        delta = current - self.last_time
        if delta > 0:
            self.fps = 1 / delta
        self.last_time = current
    
    def calculate_motion(self, landmark):

        if self.previous_landmark is None:
            self.previous_landmark = landmark.copy()
            return 0.0

        if landmark.shape != self.previous_landmark.shape:
            self.previous_landmark = landmark.copy()
            return 0.0

        motion = np.mean(
            np.abs(
                landmark -
                self.previous_landmark
            )
        )

        self.previous_landmark = landmark.copy()
        return float(motion)
    
    def get_motion_average(self, motion):
        self.motion_buffer.append(motion)
        return np.mean(self.motion_buffer)
    
    def update_motion_state(self):
        """
        Mengupdate counter gerakan.
        """
        if self.motion_score >= self.motion_high:
            self.motion_counter += 1
            self.still_counter = 0
        elif self.motion_score <= self.motion_low:
            self.still_counter += 1
            self.motion_counter = 0
        else:

            self.motion_counter = 0
            self.still_counter = 0
    
    def reset_prediction_state(self):
        """
        Reset seluruh state prediksi ketika
        berpindah mode atau tangan hilang.
        """

        self.stable_prediction = ""
        self.stable_confidence = 0.0
        self.stable_mode = ""
        self.motion_counter = 0
        self.still_counter = 0
        self.inference.reset()
        self.cnn_buffer.clear()
        self.lstm_buffer.clear()
    
    def reset_tracking(self):
        """
        Reset tracking ketika tangan hilang.
        """
        self.state = "IDLE"
        self.previous_landmark = None
        self.motion_score = 0.0
        self.motion_buffer.clear()
        self.reset_prediction_state()
    
    def sync_result(self):
        """
        Sinkronisasi hasil inferensi agar dapat
        dibaca oleh Streamlit.
        """
        with self._lock:
            self.last_prediction = self.stable_prediction
            self.result_mode = self.stable_mode
            self.confidence = self.stable_confidence

    def should_switch_to_lstm(self):
        return (
            self.motion_counter >=
            self.motion_trigger
        )
    
    def should_switch_to_cnn(self):
        return (
            self.still_counter >=
            self.still_trigger
        )
    
    def stabilize_prediction(
        self,
        label,
        confidence,
        mode,
        buffer
    ):
        if label in ("", None, "Unknown"):
            return

        buffer.append(
            (
                label,
                confidence,
                mode
            )
        )

        if len(buffer) < buffer.maxlen:
            return

        labels = [x[0] for x in buffer]

        most_common = Counter(
            labels
        ).most_common(1)[0]

        if most_common[1] < 4:
            return

        best = max(
            [x for x in buffer if x[0] == most_common[0]],
            key=lambda x: x[1]
        )

        # Confidence Lock
        if best[1] >= self.stable_confidence:
            self.stable_prediction = best[0]
            self.stable_confidence = best[1]
            self.stable_mode = best[2]
    
    def run_cnn(self, results):
        landmark = extract_cnn_landmarks(results)

        label, conf = self.inference.predict_cnn(landmark)

        if (label != "Unknown" and conf >= self.conf_cnn):
            self.stabilize_prediction(
                label,
                conf,
                "ABJAD",
                self.cnn_buffer
            )
    
    def run_lstm(self, results):
        landmark = extract_landmarks(results)

        self.inference.add_landmarks(landmark)

        if self.inference.is_sequence_ready():
            label, conf = self.inference.predict_lstm()
            if (label != "Unknown" and conf >= self.conf_lstm):
                self.stabilize_prediction(
                    label,
                    conf,
                    "KOSAKATA",
                    self.lstm_buffer
                )
    
    def auto_detect(self, results):
        # Hitung Motion
        landmark = extract_cnn_landmarks(results)

        motion = self.calculate_motion(landmark)
        self.motion_score = self.get_motion_average(motion)

        # Update Counter
        self.update_motion_state()

        # IDLE
        if self.state == "IDLE":
            self.state = "CNN"

        # CNN
        if self.state == "CNN":
            if self.should_switch_to_lstm():
                self.state = "LSTM"
                self.reset_prediction_state()
                return
            self.run_cnn(results)

        # LSTM
        elif self.state == "LSTM":
            if self.should_switch_to_cnn():
                self.state = "CNN"
                self.reset_prediction_state()
                return
            self.run_lstm(results)
    
    def draw_overlay(self, frame):
        """
        Menggambar seluruh informasi inferensi
        ke dalam frame kamera.
        """
        cv2.putText(
            frame,
            f"{self.stable_mode}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            self.stable_prediction,
            (20,75),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            2
        )

        cv2.putText(
            frame,
            f"{self.stable_confidence:.2f}",
            (20,110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,0),
            2
        )

        cv2.putText(
            frame,
            f"Motion : {self.motion_score:.4f}",
            (20,145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,255),
            2
        )

        cv2.putText(
            frame,
            f"State : {self.state}",
            (20,180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,255),
            2
        )

        cv2.putText(
            frame,
            f"FPS : {self.fps:.1f}",
            (20,215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

    def process_frame(self, frame):
        self._frame_index += 1
        do_full_process = (self._frame_index % FRAME_SKIP == 0)

        if do_full_process:
            # Downscale sebelum masuk Holistic. Landmark MediaPipe
            # bersifat koordinat ternormalisasi (0-1), jadi tetap
            # valid digambar di frame resolusi asli selama aspect
            # ratio dipertahankan.
            h, w = frame.shape[:2]
            if w > PROCESS_WIDTH:
                scale = PROCESS_WIDTH / w
                small_frame = cv2.resize(
                    frame,
                    (PROCESS_WIDTH, int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                small_frame = frame

            results = self.detector.process(small_frame)
            self._last_results = results

            if has_hand(results):
                self.auto_detect(results)
            else:
                self.reset_tracking()

            self.sync_result()

        # Gambar landmark pakai hasil deteksi terakhir yang tersedia
        # (baik dari frame ini atau frame sebelumnya yang di-skip).
        if self._last_results is not None:
            draw_landmarks(frame, self._last_results)

        self.draw_overlay(frame)

        return frame

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        image = self.process_frame(image)
        self.calculate_fps()

        return av.VideoFrame.from_ndarray(image, format="bgr24")
    
    def get_result(self):
        """
        Mengembalikan hasil inferensi terbaru
        untuk ditampilkan di Streamlit.
        """
        with self._lock:
            return {
                "prediction": self.last_prediction,
                "mode": self.result_mode,
                "confidence": self.confidence,
                "motion": self.motion_score,
                "state": self.state,
                "fps": self.fps,
                "has_prediction": self.stable_prediction != "",
            }
        self.detector.close()
    
    def __del__(self):
        try:
            self.detector.close()
        except Exception:
            pass