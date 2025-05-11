import streamlit as st
from modules.api_client import call_chat_api
from modules.visualization import visualize_problem, visualize_solution

def show_chat_interface(test_mode: bool = False):

    st.title("Scheduling Chat Assistant")

    with st.expander("What can I ask?", expanded=False):
        st.markdown("""
1. *Describe a problem* – "There are 3 machines and 5 jobs …"
2. *Ask for a solution* – "Minimise makespan"
3. *Iterate* – "Now make job 4 twice as long …"
""")

    # -------- Session state -------------------------------------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # -------- Sidebar -------------------------------------------------------
    if st.sidebar.button("🗑 Clear history", key="clear_history"):
        st.session_state.chat_history.clear()
        st.experimental_rerun()

    # -------- Render history ------------------------------------------------
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("problem"):
                visualize_problem(msg["problem"])
            if msg.get("solution"):
                visualize_solution(msg["solution"])

    # -------- User input ----------------------------------------------------
    user_input = st.chat_input("Describe your scheduling problem…")
    if not user_input:
        return

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner("Thinking…"):
        api_reply = call_chat_api(user_input, test_mode=test_mode)

    if api_reply is None:
        err = "⚠️ Couldn't reach the scheduler – please try again."
        st.session_state.chat_history.append({"role": "assistant", "content": err})
        with st.chat_message("assistant"):
            st.error(err)
        return

    # -------- Show assistant answer ----------------------------------------
    with st.chat_message("assistant"):
        st.write(api_reply["response"])
        if api_reply.get("scheduling_problem"):
            visualize_problem(api_reply["scheduling_problem"])
        if api_reply.get("api_response", {}).get("status") == "success":
            visualize_solution(api_reply)

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": api_reply["response"],
            "problem": api_reply.get("scheduling_problem"),
            "solution": api_reply
            if api_reply.get("api_response", {}).get("status") == "success"
            else None,
        }
    )
