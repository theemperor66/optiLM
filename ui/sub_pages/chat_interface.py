import streamlit as st
from modules.api_client import call_chat_api
from modules.visualization import visualize_problem, visualize_solution
from typing import Dict, Optional

def last_state(hist):
    """Get the last scheduling problem state from the chat history."""
    for m in reversed(hist):
        if m["role"] == "assistant" and m.get("problem"):
            return m["problem"]
    return None

def show_chat_interface(test_mode: bool = False):

    st.title("Scheduling Chat Assistant")

    with st.expander("What can I ask?", expanded=False):
        st.markdown("""
1. *Describe a problem* – "There are 3 machines and 5 jobs …"
2. *Add more details* – "Job 1 takes 3 minutes on rig 2"
3. *Ask for a solution* – "Solve this problem"
4. *Start over* – "Reset" or "Start over"
""")

    # -------- Session state -------------------------------------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # -------- Sidebar -------------------------------------------------------
    if st.sidebar.button("🗑 Reset problem", key="reset_problem"):
        # Send a reset message to the API
        with st.spinner("Resetting..."):
            api_reply = call_chat_api("reset", test_mode=test_mode)

        if api_reply is not None:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": api_reply["response"],
                "problem": api_reply.get("scheduling_problem"),
                "solution": None,
                "is_problem_complete": False
            })

        st.experimental_rerun()

    # -------- Render history ------------------------------------------------
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("problem"):
                # Add a badge if the problem is partial
                if msg.get("is_problem_complete") is False:
                    st.caption("🧩 partial problem")
                visualize_problem(msg["problem"])
            if msg.get("solution"):
                visualize_solution(msg["solution"])

    # -------- User input ----------------------------------------------------
    # Get the last problem state from the chat history
    context = last_state(st.session_state.chat_history)

    # Set the placeholder text based on whether we have a context
    placeholder = "Answer the assistant's question…" if context else "Describe your scheduling problem…"

    user_input = st.chat_input(placeholder)
    if not user_input:
        return

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner("Thinking…"):
        api_reply = call_chat_api(user_input, context=context, test_mode=test_mode)

    if api_reply is None:
        err = "⚠️ Couldn't reach the scheduler – please try again."
        st.session_state.chat_history.append({"role": "assistant", "content": err})
        with st.chat_message("assistant"):
            st.error(err)
        return

    # -------- Show assistant answer ----------------------------------------
    with st.chat_message("assistant"):
        if api_reply is not None:
            st.write(api_reply.get("response", "No response received from the API."))
        if api_reply and api_reply.get("scheduling_problem"):
            # Add a badge if the problem is partial
            if api_reply.get("is_problem_complete") is False:
                st.caption("🧩 partial problem")
            visualize_problem(api_reply.get("scheduling_problem"))
        if api_reply and api_reply.get("api_response") and api_reply.get("api_response", {}).get("status") == "success":
            visualize_solution(api_reply)

    # Ensure api_reply is not None before accessing its attributes
    if api_reply is not None:
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": api_reply["response"],
                "problem": api_reply.get("scheduling_problem"),
                "solution": api_reply
                if api_reply.get("api_response") and api_reply.get("api_response", {}).get("status") == "success"
                else None,
                "is_problem_complete": api_reply.get("is_problem_complete", False)
            }
        )
