"""
bisindo_app_lite.py
------------------------------------
Versi ringan BISINDO Translator.

Beda dengan bisindo_app.py (streamlit-webrtc / continuous WebRTC
stream), versi ini memakai snapshot periodik lewat webcam_component
(getUserMedia + canvas, bukan aiortc). Cocok untuk environment CPU
sangat terbatas (mis. Streamlit Community Cloud free tier) karena:

- Tidak ada overhead encode/decode video codec di server.
- Tidak ada proses WebRTC/ICE/aiortc sama sekali.
- Frame rate yang diproses sepenuhnya terkontrol (default ~4 fps),
  jauh lebih jarang dibanding continuous stream 15-30 fps.

Trade-off: terasa seperti "foto berturut-turut" bukan video mulus,
dan tiap frame baru = 1x rerun script Streamlit.

Jalankan dengan:
    streamlit run bisindo_app_lite.py
"""

import time
import cv2
import streamlit as st

from processor import BISINDOProcessor
from webcam_component import webcam_snapshot, data_url_to_bgr

from config import CNN_THRESHOLD, LSTM_THRESHOLD


def init_page():
    st.set_page_config(
        page_title="BISINDO Translator (Lite)",
        page_icon="🤟",
        layout="wide",
    )


def init_session():
    defaults = {
        "history": [],
        "kalimat": [],
        "current_word": "-",
        "current_mode": "-",
        "current_conf": 0.0,
        "last_seq": -1,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Processor disimpan di session_state supaya state (buffer motion,
    # buffer sequence LSTM, dst) tetap nyambung antar rerun.
    if "processor" not in st.session_state:
        st.session_state.processor = BISINDOProcessor()


def sidebar_ui():
    with st.sidebar:
        st.header("⚙️ Pengaturan")

        st.subheader("🎯 Confidence Threshold")
        conf_cnn = st.slider(
            "Confidence Abjad", 0.50, 1.00, CNN_THRESHOLD, 0.05
        )
        conf_lstm = st.slider(
            "Confidence Kosakata", 0.50, 1.00, LSTM_THRESHOLD, 0.05
        )

        st.divider()
        st.subheader("🤲 Motion Detection")
        motion_low = st.slider("Motion Low", 0.001, 0.020, 0.005, 0.001)
        motion_high = st.slider("Motion High", 0.005, 0.050, 0.015, 0.001)

        st.divider()
        st.subheader("📷 Snapshot")
        interval_ms = st.slider(
            "Interval snapshot (ms)",
            150, 1000, 250, 50,
            help="Makin besar = makin ringan CPU, tapi makin 'patah-patah'.",
        )
        cam_width = st.select_slider(
            "Lebar capture (px)", options=[320, 400, 480, 640], value=480
        )

        st.divider()
        st.caption(
            "Mode Lite: snapshot polling, bukan WebRTC continuous stream."
        )

    return {
        "conf_cnn": conf_cnn,
        "conf_lstm": conf_lstm,
        "motion_low": motion_low,
        "motion_high": motion_high,
        "interval_ms": interval_ms,
        "cam_width": cam_width,
    }


def camera_ui(settings):
    st.subheader("📷 Kamera (snapshot mode)")

    frame_data = webcam_snapshot(
        interval_ms=settings["interval_ms"],
        width=settings["cam_width"],
        quality=0.6,
        key="bisindo_cam",
    )

    result_placeholder = st.empty()

    if not frame_data or "image" not in frame_data:
        st.info("Menunggu izin & feed kamera dari browser...")
        return None

    seq = frame_data.get("seq", -1)

    # Hindari memproses ulang snapshot yang sama saat rerun terjadi
    # karena interaksi widget lain (bukan snapshot baru).
    if seq == st.session_state.last_seq:
        result = st.session_state.processor.get_result()
    else:
        st.session_state.last_seq = seq

        frame = data_url_to_bgr(frame_data["image"])
        if frame is None:
            st.warning("Gagal decode frame dari kamera.")
            return None

        processor = st.session_state.processor
        processor.update_settings(
            conf_cnn=settings["conf_cnn"],
            conf_lstm=settings["conf_lstm"],
            motion_low=settings["motion_low"],
            motion_high=settings["motion_high"],
        )

        annotated = processor.process_frame(frame)
        result = processor.get_result()

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        try:
            # Streamlit terbaru: parameter width="stretch"
            result_placeholder.image(annotated_rgb, channels="RGB", width="stretch")
        except TypeError:
            try:
                # Streamlit versi sebelumnya: use_container_width
                result_placeholder.image(
                    annotated_rgb, channels="RGB", use_container_width=True
                )
            except TypeError:
                # Fallback paling aman untuk versi Streamlit manapun
                result_placeholder.image(annotated_rgb, channels="RGB")

    return result


def result_ui(result):
    st.subheader("📋 Dashboard Deteksi")

    if result:
        pred = result["prediction"]
        mode = result["mode"]
        conf = result["confidence"]

        if pred not in ("", "-", "Unknown") and pred != st.session_state.current_word:
            st.session_state.current_word = pred
            st.session_state.current_mode = mode
            st.session_state.current_conf = conf

            st.session_state.history.append({
                "kata": pred,
                "mode": mode,
                "conf": conf,
                "time": time.strftime("%H:%M:%S"),
            })
            if len(st.session_state.history) > 100:
                st.session_state.history.pop(0)

            st.session_state.kalimat.append(pred)
            if len(st.session_state.kalimat) > 30:
                st.session_state.kalimat.pop(0)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Mode", st.session_state.current_mode)
    with c2:
        st.metric("State", result["state"] if result else "-")

    st.metric("Prediksi", st.session_state.current_word)
    st.metric("Confidence", f"{st.session_state.current_conf*100:.1f}%")
    st.progress(float(st.session_state.current_conf))

    st.divider()
    st.subheader("📝 Kalimat")
    st.info(" ".join(st.session_state.kalimat) if st.session_state.kalimat else "-")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Hapus"):
            if st.session_state.kalimat:
                st.session_state.kalimat.pop()
            if st.session_state.history:
                st.session_state.history.pop()
            st.rerun()
    with col2:
        if st.button("Reset"):
            st.session_state.history = []
            st.session_state.kalimat = []
            st.session_state.current_word = "-"
            st.session_state.current_mode = "-"
            st.session_state.current_conf = 0.0
            st.rerun()

    st.divider()
    st.subheader("🕒 Riwayat")
    if st.session_state.history:
        for item in reversed(st.session_state.history[-10:]):
            st.caption(f"[{item['time']}] {item['kata']} ({item['mode']}) {item['conf']:.2f}")
    else:
        st.caption("Belum ada riwayat.")


@st.fragment
def camera_and_result_fragment(settings):
    """
    Dibungkus st.fragment supaya tiap frame snapshot baru cuma
    me-rerun BAGIAN INI SAJA (kamera + dashboard), bukan seluruh
    halaman. Sidebar, judul, dll tidak akan ikut "refresh" tiap
    frame -> jauh lebih nyaman dilihat & sedikit lebih ringan.
    """
    left, right = st.columns([2, 1])
    with left:
        result = camera_ui(settings)
    with right:
        result_ui(result)


def main():
    init_page()
    init_session()
    settings = sidebar_ui()

    st.title("🤟 BISINDO Translator — Lite Mode")
    st.markdown(
        "Snapshot polling (tanpa WebRTC) — versi ringan untuk CPU terbatas."
    )
    st.divider()

    camera_and_result_fragment(settings)


if __name__ == "__main__":
    main()
