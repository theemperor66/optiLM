import os
import requests
import streamlit as st
from typing import Optional, Dict, Union, List
import functools

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

def call_chat_api(message: str,
                  context: Optional[Dict] = None,
                  message_history: Optional[List[Dict[str, str]]] = None,
                  test_mode: bool = False,
                  timeout: float = 120.0) -> Optional[Dict]:
    """
    Send *message* (and optional *context* and *message_history*) to the OWPy back-end.

    Args:
        message: The user message to send
        context: Optional problem state context
        message_history: Optional conversation history in the format [{'role': 'user|assistant', 'content': 'message'}]
        test_mode: Whether to use test mode
        timeout: API request timeout in seconds
        
    Returns:
        The parsed JSON response on success, or *None* and shows an error banner on failure.
    """
    payload = {"message": message, "test_mode": test_mode}
    if context is not None:
        payload["context"] = context
    if message_history is not None:
        payload["message_history"] = message_history

    try:
        r = requests.post(f"{API_URL}/chat", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()

    except requests.Timeout:
        st.error("OWPy API timed-out after 120 seconds. The server might be busy processing a complex problem. Please try again.")
    except requests.RequestException as e:
        st.error(f"OWPy API error: {e}")
    return None

@functools.lru_cache(maxsize=1)
def get_available_solvers(timeout: float = 20.0) -> List[str]:
    """
    Fetch the list of available solvers from the OWPy API.

    Returns a list of solver names on success, or a default list on failure.
    Uses LRU cache to avoid repeated API calls.
    """
    default_solvers = ["GLOBAL", "TWO_STEP", "SA", "BranchAndBound"]

    try:
        r = requests.get(f"{API_URL}/api/v1/solvers/", timeout=timeout)
        r.raise_for_status()
        solvers = r.json()
        return solvers if solvers else default_solvers
    except (requests.RequestException, ValueError) as e:
        st.warning(f"Could not fetch available solvers: {e}. Using default list.")
        return default_solvers
