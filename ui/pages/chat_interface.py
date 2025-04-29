import streamlit as st
from ..modules.api_client import call_chat_api
from ..modules.visualization import visualize_problem, visualize_solution

def show_chat_interface(test_mode=False):
    """
    Display the Chat Interface page.
    
    Args:
        test_mode (bool, optional): Whether to use test mode. Defaults to False.
    """
    st.title("Scheduling Chat Assistant")

    with st.expander("About this chat", expanded=False):
        st.markdown("""
        This chat interface allows you to describe scheduling problems in natural language.

        The assistant will:
        1. Understand your scheduling problem description
        2. Extract the machines, jobs, and rig requirements
        3. Formulate a structured scheduling problem
        4. Solve the problem using the OWPy API to minimize makespan
        5. Present the results in a user-friendly way with a Gantt chart

        Try asking something like: "I have 3 machines and 5 jobs. Each job requires a specific rig. The rig change time between rig 1 and rig 2 is 2 units."
        """)

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            # If this is an assistant message with a problem, visualize it
            if message["role"] == "assistant" and "problem" in message:
                visualize_problem(message["problem"])

            # If this is an assistant message with a solution, visualize it
            if message["role"] == "assistant" and "solution" in message:
                visualize_solution(message["solution"])

    # Chat input
    user_input = st.chat_input("Describe your scheduling problem...")

    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Display user message
        with st.chat_message("user"):
            st.write(user_input)

        # Get response from API
        with st.spinner("Thinking..."):
            api_response = call_chat_api(user_input, test_mode=test_mode)

            if api_response:
                # Display assistant response
                with st.chat_message("assistant"):
                    st.write(api_response["response"])

                    # If there's a scheduling problem, visualize it
                    if api_response.get("scheduling_problem"):
                        visualize_problem(api_response["scheduling_problem"])

                    # If there's a solution, visualize it
                    if api_response.get("api_response") and api_response["api_response"]["status"] == "success":
                        visualize_solution(api_response)

                # Add assistant message to chat history
                assistant_message = {
                    "role": "assistant", 
                    "content": api_response["response"]
                }

                # Add problem and solution to the message if available
                if api_response.get("scheduling_problem"):
                    assistant_message["problem"] = api_response["scheduling_problem"]

                if api_response.get("api_response") and api_response["api_response"]["status"] == "success":
                    assistant_message["solution"] = api_response

                st.session_state.chat_history.append(assistant_message)
            else:
                # Display error message
                with st.chat_message("assistant"):
                    st.error("Sorry, I couldn't process your request. Please try again later.")

                # Add error message to chat history
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": "Sorry, I couldn't process your request. Please try again later."
                })