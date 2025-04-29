import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import pages
from pages.problem_builder import show_problem_builder
from pages.chat_interface import show_chat_interface

# Page configuration
st.set_page_config(
    page_title="OWPy Scheduling Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Scheduling Problem Builder", "Chat Interface"])

# Test mode toggle
st.sidebar.title("Settings")
test_mode = st.sidebar.checkbox("Test Mode (Random Solutions)", value=False, 
                               help="When enabled, the system will generate random solutions instead of calling the OWPy API.")

# Display the selected page
if page == "Scheduling Problem Builder":
    show_problem_builder(test_mode=test_mode)
elif page == "Chat Interface":
    show_chat_interface(test_mode=test_mode)

# Footer
st.markdown("---")
st.markdown("OWPy Scheduling Assistant | Built with Streamlit")
