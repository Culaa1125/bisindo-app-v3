"""
detectors.py
----------------------------------------
MediaPipe Holistic Manager
Landmark Extraction
"""

import threading
import cv2
import mediapipe as mp
import numpy as np

from config import (
    MODEL_COMPLEXITY,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    ENABLE_SEGMENTATION,
    SMOOTH_SEGMENTATION,
    REFINE_FACE_LANDMARKS,
    SMOOTH_LANDMARKS,
    CNN_INPUT_SIZE,
    LANDMARK_VECTOR_SIZE,
)

# =====================================================
# MediaPipe Drawing
# =====================================================

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


class HolisticManager:
    """
    Mengelola MediaPipe Holistic.
    """
    def __init__(self):
        self.holistic = None
        self.lock = threading.Lock()

    def get_holistic(self):
        if self.holistic is None:
            self.holistic = mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=MODEL_COMPLEXITY,
                smooth_landmarks=SMOOTH_LANDMARKS,
                enable_segmentation=ENABLE_SEGMENTATION,
                smooth_segmentation=SMOOTH_SEGMENTATION,
                refine_face_landmarks=REFINE_FACE_LANDMARKS,
                min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            )
        return self.holistic

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        holistic = self.get_holistic()
        with self.lock:
            results = holistic.process(rgb)
        rgb.flags.writeable = True
        return results

    def close(self):
        if self.holistic is not None:
            self.holistic.close()
            self.holistic = None

# =====================================================
# LANDMARK EXTRACTION
# =====================================================

def extract_landmarks(results):
    """
    Menghasilkan vector 258 fitur
    """
    # -------------------------
    # Pose
    # -------------------------
    if results.pose_landmarks:
        pose = np.array([
            [
                lm.x,
                lm.y,
                lm.z,
                lm.visibility
            ]
            for lm in results.pose_landmarks.landmark
        ]).flatten()
    else:
        pose = np.zeros(
        33*4,
        dtype=np.float32
    )

    # -------------------------
    # Left Hand
    # -------------------------
    if results.left_hand_landmarks:
        left = np.array([
            [
                lm.x,
                lm.y,
                lm.z
            ]
            for lm in results.left_hand_landmarks.landmark
        ]).flatten()
    else:
        left = np.zeros(
            21 * 3,
            dtype=np.float32
        )

    # -------------------------
    # Right Hand
    # -------------------------
    if results.right_hand_landmarks:
        right = np.array([
            [
                lm.x,
                lm.y,
                lm.z
            ]
            for lm in results.right_hand_landmarks.landmark
        ]).flatten()
    else:
        right = np.zeros(
            21 * 3,
            dtype=np.float32
        )

    return np.concatenate(
        [pose, left, right]
    ).astype(np.float32)

def extract_cnn_landmarks(results):
    """
    Menghasilkan 126 fitur tangan dari hasil Holistic.
    Format identik dengan preprocessing dataset CNN.
    """
    lm = np.zeros(CNN_INPUT_SIZE, dtype=np.float32)

    # LEFT HAND
    if results.left_hand_landmarks:
        for i, pt in enumerate(results.left_hand_landmarks.landmark):
            lm[i*3] = pt.x
            lm[i*3+1] = pt.y
            lm[i*3+2] = pt.z

    # RIGHT HAND
    offset = 63
    if results.right_hand_landmarks:
        for i, pt in enumerate(results.right_hand_landmarks.landmark):
            lm[offset+i*3] = pt.x
            lm[offset+i*3+1] = pt.y
            lm[offset+i*3+2] = pt.z

    return lm

def draw_hand_landmarks(frame, results):
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

def draw_landmarks(frame, results):
    """
    Menggambar landmark ke frame.
    """
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
        )

    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
        )

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
        )

def has_hand(results):
    return (
        results.left_hand_landmarks is not None
        or
        results.right_hand_landmarks is not None
    )
