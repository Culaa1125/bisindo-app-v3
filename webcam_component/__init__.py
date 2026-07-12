"""
webcam_component
------------------------------------
Komponen kamera ringan berbasis snapshot periodik (bukan WebRTC
continuous stream). Dibuat manual dengan HTML/JS polos (tanpa
build tools/npm) memakai protokol Streamlit Components bawaan.

Cara pakai:

    from webcam_component import webcam_snapshot

    frame = webcam_snapshot(interval_ms=250, width=480, quality=0.6)
    if frame and "image" in frame:
        # frame["image"] adalah data URL "data:image/jpeg;base64,...."
        ...
"""

import os
import base64
import numpy as np
import cv2
import streamlit.components.v1 as components

_COMPONENT_DIR = os.path.dirname(os.path.abspath(__file__))

_component_func = components.declare_component(
    "bisindo_webcam_snapshot",
    path=_COMPONENT_DIR,
)


def webcam_snapshot(interval_ms=250, width=480, quality=0.6, key=None):
    """
    Menampilkan widget kamera dan mengembalikan dict hasil snapshot
    terakhir dari browser:

        {"image": "data:image/jpeg;base64,...", "seq": int,
         "width": int, "height": int, "ts": int}

    atau None jika belum ada frame yang masuk.
    """
    return _component_func(
        interval_ms=interval_ms,
        width=width,
        quality=quality,
        key=key,
        default=None,
    )


def data_url_to_bgr(data_url):
    """
    Mengonversi data URL base64 (dari webcam_snapshot) menjadi
    array numpy BGR (format yang dipakai OpenCV/MediaPipe).
    Return None jika data_url tidak valid.
    """
    if not data_url or "," not in data_url:
        return None

    try:
        header, b64data = data_url.split(",", 1)
        img_bytes = base64.b64decode(b64data)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None
