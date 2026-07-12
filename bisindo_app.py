import time
import streamlit as st

from streamlit_webrtc import webrtc_streamer

from processor import BISINDOProcessor

from config import (
    CNN_THRESHOLD,
    LSTM_THRESHOLD,
    MAX_HISTORY,
)

from rtc_config import RTC_CONFIGURATION

def init_page():
    st.set_page_config(
        page_title="BISINDO Translator",
        page_icon="🤟",
        layout="wide"
    )

def init_session():
    defaults = {
        "history": [],
        "kalimat": [],
        "current_word": "-",
        "current_mode": "-",
        "current_conf": 0.0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def sidebar_ui():
    """
    Sidebar pengaturan aplikasi.
    Mengembalikan seluruh konfigurasi yang akan
    dikirim ke BISINDOProcessor.
    """

    with st.sidebar:

        st.header("⚙️ Pengaturan")

        # ============================================
        # CONFIDENCE
        # ============================================

        st.subheader("🎯 Confidence Threshold")

        conf_cnn = st.slider(
            "Confidence Abjad",
            min_value=0.50,
            max_value=1.00,
            value=CNN_THRESHOLD,
            step=0.05,
            help="Confidence minimum untuk prediksi huruf."
        )

        conf_lstm = st.slider(
            "Confidence Kosakata",
            min_value=0.50,
            max_value=1.00,
            value=LSTM_THRESHOLD,
            step=0.05,
            help="Confidence minimum untuk prediksi kosakata."
        )

        st.divider()

        # ============================================
        # MOTION
        # ============================================

        st.subheader("🤲 Motion Detection")

        motion_low = st.slider(
            "Motion Low",
            min_value=0.001,
            max_value=0.020,
            value=0.005,
            step=0.001,
            help="Batas perpindahan kembali ke mode Abjad."
        )

        motion_high = st.slider(
            "Motion High",
            min_value=0.005,
            max_value=0.050,
            value=0.015,
            step=0.001,
            help="Batas perpindahan ke mode Kosakata."
        )

        st.divider()

        # ============================================
        # INFO
        # ============================================

        st.subheader("📊 Model")

        st.caption("CNN : 126 Landmark")
        st.caption("BiLSTM : 30 × 258 Landmark")
        st.caption("Deteksi : Otomatis")

        st.divider()

        st.success("Success: Automatic CNN ↔ BiLSTM Switching")

        st.divider()

        st.subheader("🌐 Koneksi WebRTC")

        has_turn = any(
            any("turn:" in url or "turns:" in url for url in server.get("urls", []))
            for server in RTC_CONFIGURATION["iceServers"]
        )

        if has_turn:
            st.success("Success: TURN Server Aktif (Twilio)")
        else:
            st.warning("Warning: Hanya menggunakan STUN")
            twilio_error = st.session_state.get("_twilio_error")
            if twilio_error:
                st.caption(f"Twilio error: {twilio_error}")

    return {
        "conf_cnn": conf_cnn,
        "conf_lstm": conf_lstm,
        "motion_low": motion_low,
        "motion_high": motion_high,
    }

#st.write(RTC_CONFIGURATION)

def camera_ui(settings):
    """
    Menampilkan kamera dan mengirimkan
    seluruh konfigurasi ke BISINDOProcessor.
    """
    st.subheader("📷 Kamera Real-Time")

    ctx = webrtc_streamer(
        key="bisindo",
        video_processor_factory=BISINDOProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        # True: recv() jalan di thread terpisah dari loop utama WebRTC,
        # supaya frame yang masuk tidak nge-block/menumpuk selagi
        # menunggu inferensi selesai (penting di CPU terbatas).
        async_processing=True,
        media_stream_constraints={
            "video": {
                "width": 640,
                "height": 480,
                # 10 fps cukup untuk isyarat tangan dan mengurangi
                # jumlah frame yang perlu didekode aiortc di server.
                "frameRate": 10,
            },
            "audio": False,
        },
    )

    if ctx and ctx.video_processor:
        ctx.video_processor.update_settings(
            conf_cnn=settings["conf_cnn"],
            conf_lstm=settings["conf_lstm"],
            motion_low=settings["motion_low"],
            motion_high=settings["motion_high"],
        )

    return ctx

def result_ui(ctx):
    st.subheader("📋 Dashboard Deteksi")

    # ============================
    # Ambil hasil dari processor
    # ============================
    if ctx and ctx.video_processor:

        result = ctx.video_processor.get_result()

        pred = result["prediction"]
        mode = result["mode"]
        conf = result["confidence"]
        motion = result["motion"]
        state = result["state"]
        fps = result["fps"]

        if pred not in ("", "-", "Unknown"):

            if pred != st.session_state.current_word:

                st.session_state.current_word = pred
                st.session_state.current_mode = mode
                st.session_state.current_conf = conf

                st.session_state.history.append({
                    "kata": pred,
                    "mode": mode,
                    "conf": conf,
                    "time": time.strftime("%H:%M:%S")
                })

                if len(st.session_state.history) > 100:
                    st.session_state.history.pop(0)

                st.session_state.kalimat.append(pred)

                if len(st.session_state.kalimat) > 30:
                    st.session_state.kalimat.pop(0)

    # ============================================
    # Dashboard
    # ============================================

    st.markdown("### 📌 Informasi Real-Time")

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Mode",
            st.session_state.current_mode
        )

    with c2:
        st.metric(
            "State",
            result["state"] if ctx and ctx.video_processor else "-"
        )

    st.metric(
        "Prediksi",
        st.session_state.current_word
    )

    st.metric(
        "Confidence",
        f"{st.session_state.current_conf*100:.1f}%"
    )

    st.progress(
        float(st.session_state.current_conf)
    )

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Motion",
            f"{motion:.4f}" if ctx and ctx.video_processor else "-"
        )

    with c2:
        st.metric(
            "FPS",
            f"{fps:.1f}" if ctx and ctx.video_processor else "-"
        )

    st.divider()

    st.subheader("📝 Kalimat")

    if st.session_state.kalimat:
        st.info(
            " ".join(
                st.session_state.kalimat
            )
        )
    else:
        st.info("-")

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
        for item in reversed(
            st.session_state.history[-10:]
        ):
            st.caption(
                f"[{item['time']}] "
                f"{item['kata']} "
                f"({item['mode']}) "
                f"{item['conf']:.2f}"
            )

    else:
        st.caption("Belum ada riwayat.")
    
def footer_ui():

    st.divider()

    with st.expander("📖 Informasi Sistem"):

        st.markdown("""

### 🤟 BISINDO Translator

Aplikasi ini menerapkan sistem pengenalan Bahasa Isyarat Indonesia (BISINDO)
secara **real-time** menggunakan kombinasi dua model Deep Learning.

---

### 🔤 CNN (Abjad)

- Digunakan untuk mengenali huruf statis.
- Input berupa **126 landmark tangan**.
- Menggunakan model Fully Connected Neural Network.
- Prediksi dilakukan setiap frame.

---

### 🤲 BiLSTM (Kosakata)

- Digunakan untuk mengenali gerakan dinamis.
- Input berupa **30 frame landmark pose + tangan (258 fitur)**.
- Menggunakan Bidirectional Long Short-Term Memory (BiLSTM).
- Cocok untuk mengenali kata atau gerakan berurutan.

---

### 🔄 Automatic Switching

Sistem secara otomatis menentukan model yang digunakan berdasarkan
besar kecilnya pergerakan landmark tangan.

- Motion rendah → CNN (Abjad)
- Motion tinggi → BiLSTM (Kosakata)

---

### 🌐 Real-Time Communication

Aplikasi menggunakan:

- Streamlit WebRTC
- MediaPipe Holistic
- TensorFlow
- STUN / TURN Server (Metered)

sehingga dapat berjalan secara real-time baik pada jaringan lokal maupun saat
dideploy ke Streamlit Cloud.

        """)

def main():
    init_page()
    init_session()
    settings = sidebar_ui()
    st.title("🤟 BISINDO Translator")
    st.markdown(
        "Automatic CNN + BiLSTM"
    )
    st.divider()
    left, right = st.columns([2,1])
    with left:
        ctx = camera_ui(settings)
        if ctx and ctx.state.playing:
            st.success("Success: Kamera Aktif")
        else:
            st.warning("Klik START untuk mengaktifkan kamera.")
    with right:
        result_ui(ctx)
    with st.expander("RTC Debug"):
        st.json(RTC_CONFIGURATION)
    footer_ui()

if __name__ == "__main__":
    main()
