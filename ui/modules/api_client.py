import requests
import streamlit as st
import os

# API URL (configurable via environment variable)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

def call_chat_api(message, context=None, test_mode=False):
    """
    Call the chat API with the given message and context.
    
    Args:
        message (str): The message to send to the API
        context (dict, optional): Additional context for the API. Defaults to None.
        test_mode (bool, optional): Whether to use test mode. Defaults to False.
        
    Returns:
        dict: The API response, or None if there was an error
    """
    try:
        payload = {
            "message": message,
            "test_mode": test_mode
        }
        if context:
            payload["context"] = context

        response = requests.post(f"{API_URL}/chat", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None