---
title: BISINDO Translator
emoji: 🤟
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.39.0"
app_file: bisindo_app.py
pinned: false
python_version: "3.10"
---

# 🤟 BISINDO Translator

Aplikasi penerjemah Bahasa Isyarat Indonesia (BISINDO) real-time menggunakan
kombinasi model **CNN** (abjad) dan **BiLSTM** (kosakata), dengan input
kamera langsung via WebRTC.

## Dua Mode Aplikasi

| File | Mode | Cocok untuk |
|---|---|---|
| `bisindo_app.py` | WebRTC continuous stream (streamlit-webrtc) | CPU/RAM cukup lega (HF PRO Docker, VPS, dst) |
| `bisindo_app_lite.py` | Snapshot polling ringan (tanpa WebRTC/aiortc) | CPU sangat terbatas (Streamlit Cloud free tier) |

`bisindo_app.py` sudah dioptimasi (`model_complexity=0`, frame-skip,
downscale sebelum MediaPipe, batas thread TensorFlow) supaya lebih
ringan dari versi awal, tapi tetap pakai WebRTC sehingga ada overhead
encode/decode video di server.

`bisindo_app_lite.py` mengambil snapshot kamera tiap beberapa ratus
milidetik lewat komponen HTML/JS polos (`webcam_component/`), lalu
memproses satu frame per rerun Streamlit — tanpa `aiortc`, tanpa
codec video. Jauh lebih ringan di CPU, dengan trade-off tampilan
terasa "per-langkah" bukan video mengalir mulus, dan mode
kosakata (BiLSTM, butuh 30 frame berurutan) akan butuh beberapa
detik untuk terisi penuh tergantung interval snapshot yang dipilih.

Untuk deploy ke Streamlit Cloud / HF Spaces, tentukan `app_file` yang
sesuai (di HF, ubah field `app_file` di frontmatter README ini; di
Streamlit Cloud, pilih file utama saat deploy).

## Cara Kerja

- **CNN** mengenali huruf statis dari 126 landmark tangan.
- **BiLSTM** mengenali gerakan/kosakata dinamis dari 30 frame x 258 fitur
  landmark pose + tangan.
- Sistem otomatis berpindah mode berdasarkan besar kecilnya pergerakan
  tangan (motion detection).

## Konfigurasi TURN Server (opsional, disarankan)

Karena Space ini berjalan di jaringan cloud, sebagian pengguna dengan
jaringan yang ketat (kampus/kantor/mobile data) mungkin butuh **TURN
server** agar WebRTC berhasil connect (STUN saja kadang tidak cukup).

Aplikasi ini menggunakan [Twilio Network Traversal Service](https://www.twilio.com/docs/stun-turn)
untuk mendapatkan kredensial TURN secara dinamis (lebih stabil daripada
TURN statis, karena Twilio otomatis memberi beberapa opsi server TURN
UDP/TCP/TLS dan credential yang selalu fresh/tidak pernah expired saat
dipakai).

1. Buat akun di [twilio.com](https://www.twilio.com/) (ada free trial
   dengan kuota TURN gratis).
2. Ambil **Account SID** dan **Auth Token** dari Twilio Console.
3. Tambahkan **Repository secrets** berikut di menu
   *Settings > Variables and secrets* pada Space ini (atau di
   `.streamlit/secrets.toml` untuk Streamlit Cloud):

| Secret Name           | Contoh Nilai                     |
|------------------------|-----------------------------------|
| `TWILIO_ACCOUNT_SID`   | `ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx` |
| `TWILIO_AUTH_TOKEN`    | `your_auth_token`                 |
| `STUN_URL` (opsional)  | `stun:stun.l.google.com:19302`    |

Tanpa secrets ini, aplikasi tetap berjalan menggunakan STUN publik Google
sebagai fallback (mungkin gagal connect di sebagian jaringan yang ketat,
termasuk saat diakses dari Streamlit Community Cloud).
