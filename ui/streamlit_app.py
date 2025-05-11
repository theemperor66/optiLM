import os, sys
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# -- Bootstrapping -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:          # <= NEW
    sys.path.insert(0, str(PROJECT_ROOT))      # <= NEW
load_dotenv(PROJECT_ROOT / ".env")             # reads .env beside ui/

# -- Internal imports (keep AFTER the path tweak) ----------------------------
from sub_pages.problem_builder import show_problem_builder
from sub_pages.chat_interface import show_chat_interface

# -- Streamlit layout --------------------------------------------------------
st.set_page_config(
    page_title="OWPy Scheduling Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("Navigation")
view = st.sidebar.radio("Go to", ["Scheduling Problem Builder", "Chat Interface"])

st.sidebar.title("Settings")
test_mode = st.sidebar.checkbox(
    "Test Mode (Random Solutions)",
    value=False,
    help=("When enabled, the system will generate random solutions "
          "instead of calling the OWPy API.")
)

if view == "Scheduling Problem Builder":
    show_problem_builder(test_mode=test_mode)
else:
    show_chat_interface(test_mode=test_mode)

st.markdown("---")
st.caption("OWPy Scheduling Assistant  |  Built with Streamlit")
