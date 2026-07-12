"""
rtc_config.py
------------------------------------
RTC Configuration
Mendukung:
- Lokal
- Streamlit Cloud
- Hugging Face Spaces
- Twilio TURN (Network Traversal Service)
"""

import os
import requests
import streamlit as st

# ======================================================
# DEFAULT STUN (fallback)
# ======================================================
DEFAULT_STUN = "stun:stun.l.google.com:19302"

# Twilio Token API expired dalam 24 jam (86400 detik),
# kita refresh sedikit lebih awal (55 menit) supaya aman.
TWILIO_TOKEN_TTL = 3300


def _get_secret(key):
    """
    Ambil secret dari st.secrets (Streamlit Cloud, via
    .streamlit/secrets.toml) jika tersedia, jika tidak
    fallback ke environment variable (Hugging Face Spaces
    "Repository secrets" diekspos sebagai env var).
    """
    try:
        # st.secrets akan raise error jika file secrets.toml
        # tidak ada sama sekali (bukan hanya KeyError), jadi
        # dibungkus try/except Exception di sini.
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    return os.environ.get(key)


# ======================================================
# LOAD SECRETS
# ======================================================
STUN_URL = _get_secret("STUN_URL") or DEFAULT_STUN
TWILIO_ACCOUNT_SID = _get_secret("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = _get_secret("TWILIO_AUTH_TOKEN")


@st.cache_data(ttl=TWILIO_TOKEN_TTL, show_spinner=False)
def _fetch_twilio_ice_servers(account_sid, auth_token):
    """
    Minta kredensial TURN/STUN sementara ke Twilio Network
    Traversal Service. Twilio akan mengembalikan daftar
    ice_servers (STUN + beberapa TURN, UDP/TCP/TLS) yang
    sudah siap pakai langsung sebagai iceServers.

    Docs: https://www.twilio.com/docs/stun-turn
    """
    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Tokens.json"

    response = requests.post(
        url,
        auth=(account_sid, auth_token),
        timeout=10,
    )
    response.raise_for_status()

    data = response.json()
    return data.get("ice_servers", [])


def _get_ice_servers():
    """
    Bangun daftar iceServers:
    - Jika kredensial Twilio tersedia -> pakai TURN dari Twilio.
    - Jika gagal / tidak ada kredensial -> fallback ke STUN saja.
    """
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
        try:
            twilio_servers = _fetch_twilio_ice_servers(
                TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN
            )
            if twilio_servers:
                return twilio_servers
        except Exception as e:
            # Jangan sampai error koneksi ke Twilio bikin app crash,
            # cukup fallback ke STUN dan tampilkan warning di UI.
            st.session_state["_twilio_error"] = str(e)

    return [{"urls": [STUN_URL]}]


ice_servers = _get_ice_servers()

RTC_CONFIGURATION = {
    "iceServers": ice_servers,
    "iceTransportPolicy": "all",
}
