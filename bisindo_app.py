import time
import streamlit as st

from streamlit_webrtc import webrtc_streamer

# ====================================================================
# HACK / PATCH DINAMIS UNTUK BUG AIOICE DI STREAMLIT CLOUD
# Mengatasi error: AttributeError: 'NoneType' object has no attribute 'sendto'
# ====================================================================
import aioice.ice

# Kita cari secara otomatis class apa yang memiliki method 'send_stun'
for name, obj in vars(aioice.ice).items():
    # Jika dia adalah sebuah Class dan punya fungsi 'send_stun'
    if isinstance(obj, type) and hasattr(obj, "send_stun"):
        _original_send_stun = getattr(obj, "send_stun")
        
        # Buat penambalnya
        def _make_patched(orig_func):
            def _patched(self, message, addr):
                # Jika Streamlit Cloud memblokir socket (transport = None), abaikan!
                if getattr(self, "transport", None) is None:
                    return
                return orig_func(self, message, addr)
            return _patched

        # Timpa fungsi aslinya dengan fungsi kebal kita
        setattr(obj, "send_stun", _make_patched(_original_send_stun))
# ====================================================================

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

        def _server_urls(server):
            """
            Normalisasi field 'urls': bisa berupa string tunggal
            (format Twilio) atau list (format lama/manual).
            """
            urls = server.get("urls", [])
            if isinstance(urls, str):
                return [urls]
            return urls

        has_turn = any(
            any("turn:" in url or "turns:" in url for url in _server_urls(server))
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

    # ============================================
    # 1. BUAT WADAH KOSONG (PLACEHOLDERS)
    # ============================================
    # Wadah ini disiapkan di luar loop agar komponen 
    # UI tidak di-render ulang secara berantakan (flicker).
    
    st.markdown("### 📌 Informasi Real-Time")
    c1, c2 = st.columns(2)
    with c1:
        mode_placeholder = st.empty()
    with c2:
        state_placeholder = st.empty()

    pred_placeholder = st.empty()
    conf_placeholder = st.empty()
    prog_placeholder = st.empty()

    st.divider()

    c3, c4 = st.columns(2)
    with c3:
        motion_placeholder = st.empty()
    with c4:
        fps_placeholder = st.empty()

    st.divider()
    st.subheader("📝 Kalimat")
    kalimat_placeholder = st.empty()

    # Tombol aksi tetap berada di luar loop agar ter-render statis
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Hapus"):
            if st.session_state.kalimat: st.session_state.kalimat.pop()
            if st.session_state.history: st.session_state.history.pop()
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
    history_placeholder = st.empty()

    # ============================================
    # 2. LOOP REAL-TIME UNTUK UPDATE DASHBOARD
    # ============================================
    if ctx and ctx.state.playing:
        import time # Pastikan module time sudah di-import di atas
        
        while True:
            if ctx.video_processor:
                result = ctx.video_processor.get_result()

                pred = result["prediction"]
                mode = result["mode"]
                conf = result["confidence"]
                motion = result["motion"]
                state = result["state"]
                fps = result["fps"]

                # Cek histori kalimat
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

                # Suntikkan data terbaru ke wadah yang sudah dibuat tadi
                mode_placeholder.metric("Mode", st.session_state.current_mode)
                state_placeholder.metric("State", state)
                pred_placeholder.metric("Prediksi", st.session_state.current_word)
                conf_placeholder.metric("Confidence", f"{st.session_state.current_conf*100:.1f}%")
                prog_placeholder.progress(float(st.session_state.current_conf))

                motion_placeholder.metric("Motion", f"{motion:.4f}")
                fps_placeholder.metric("FPS", f"{fps:.1f}")

                if st.session_state.kalimat:
                    kalimat_placeholder.info(" ".join(st.session_state.kalimat))
                else:
                    kalimat_placeholder.info("-")

                # Format ulang riwayat teks
                if st.session_state.history:
                    history_text = ""
                    for item in reversed(st.session_state.history[-10:]):
                        history_text += f"[{item['time']}] {item['kata']} ({item['mode']}) {item['conf']:.2f}\n\n"
                    history_placeholder.caption(history_text)
                else:
                    history_placeholder.caption("Belum ada riwayat.")

            # Beri jeda 100ms agar loop ini tidak mencekik 100% resource CPU
            time.sleep(0.1)
    
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
