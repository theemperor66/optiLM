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
### Basic Commands
1. *Describe a problem* – "There are 3 machines and 5 jobs …"
2. *Add more details* – "Job 1 takes 3 minutes on rig 2"
3. *Ask for a solution* – "Solve this problem"
4. *Start over* – "Reset" or "Start over"

### Key Terms Explained
- **Machine**: A resource that processes jobs (e.g., an oven, a printer, a production line)
- **Job**: A task that needs to be scheduled (e.g., baking a cake, printing a document)
- **Rig**: A configuration or setup of a machine (e.g., temperature setting, tool attachment)
- **Rig Change Time**: Time required to switch from one rig to another
- **Processing Time**: Time required to complete a job

### Sample Dialogue
**User**: "I have a bakery with 2 ovens and need to bake 5 different items."

**Assistant**: "I've added 2 machines. Can you tell me about the jobs?"

**User**: "I need to bake bread at 350°F for 30 minutes, cookies at 375°F for 12 minutes, and a cake at 325°F for 45 minutes."

**Assistant**: "I've added those jobs. Can you tell me how long it takes to change the oven temperature?"

**User**: "It takes about 5 minutes to change the temperature by 25 degrees."

**Assistant**: "Thanks! I've added the rig change times. Would you like me to solve this scheduling problem now?"
""")

    # -------- Session state -------------------------------------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # TODO: Future Enhancement - Example Library
    # This is where an example library could be implemented in the future.
    # The library would include pre-built examples for common scheduling scenarios:
    # - Manufacturing examples (machines with different tools/rigs)
    # - Service industry examples (staff scheduling)
    # - Food production examples (bakery, restaurant kitchen)
    # Each example would have a button to load it into the chat interface.

    # TODO: Future Enhancement - Guided Problem Creation
    # Implement a wizard-style interface for step-by-step problem creation:
    # 1. Add structured forms as an alternative to free-text input
    # 2. Create a multi-step wizard with separate sections for:
    #    - Defining machines
    #    - Adding jobs
    #    - Specifying rig change times
    #    - Setting solver parameters
    # 3. Provide templates for common scheduling scenarios
    # 4. Add visual cues about what information is still needed
    # 5. Show progress indicators for problem completion

    # TODO: Future Enhancement - Long-term Improvements
    # 1. Interactive Tutorial:
    #    - Create a guided walkthrough for first-time users
    #    - Include interactive exercises to practice building problems
    #    - Provide feedback and hints during the tutorial
    #
    # 2. "Learn by Example" Mode:
    #    - Show both natural language descriptions and resulting JSON structures
    #    - Demonstrate how changes to descriptions affect problem formulation
    #    - Allow users to experiment with modifications to examples
    #
    # 3. Progressive Complexity:
    #    - Allow users to specify their expertise level
    #    - Adjust UI complexity and explanations based on user level
    #    - Gradually introduce advanced features as users become more comfortable

    # -------- Sidebar -------------------------------------------------------
    if st.sidebar.button("🗑 Reset problem", key="reset_problem"):
        # Send a reset message to the API
        with st.spinner("Resetting..."):
            api_reply = call_chat_api("reset", test_mode=test_mode)

        if api_reply is not None:
            # Clear the chat history before adding the new message
            st.session_state.chat_history = [{
                "role": "assistant",
                "content": api_reply["response"],
                "problem": api_reply.get("scheduling_problem"),
                "solution": None,
                "is_problem_complete": False
            }]

        st.rerun()

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
