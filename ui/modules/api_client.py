import os
import requests
import streamlit as st
from typing import Optional, Dict, Union

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

def call_chat_api(message: str,
                  context: Optional[Dict] = None,
                  test_mode: bool = False,
                  timeout: float = 15.0) -> Optional[Dict]:
    """
    Send *message* (and optional *context*) to the OWPy back-end.

    Returns the parsed JSON on success, or *None* and shows an error
    banner on failure.
    """
    payload = {"message": message, "test_mode": test_mode}
    if context is not None:
        payload["context"] = context

    try:
        r = requests.post(f"{API_URL}/chat", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()

    except requests.Timeout:
        st.error("OWPy API timed-out – try again or increase the timeout.")
    except requests.RequestException as e:
        st.error(f"OWPy API error: {e}")
    return None
