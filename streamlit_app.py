import streamlit as st
import requests
import json
import plotly.figure_factory as ff
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API URL (configurable via environment variable)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

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

# Function to call the chat API
def call_chat_api(message, context=None, test_mode=False):
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

# Function to visualize the scheduling problem
def visualize_problem(problem):
    st.subheader("Scheduling Problem Visualization")

    # Display machines
    st.write("**Machines:**")
    if 'machines' in problem:
        machines_df = pd.DataFrame(problem['machines'])
        st.dataframe(machines_df)
    else:
        st.warning("No machine information available")

    # Display jobs
    st.write("**Jobs:**")
    if 'jobs' in problem:
        jobs_df = pd.DataFrame(problem['jobs'])
        st.dataframe(jobs_df)
    else:
        st.warning("No job information available")

    # Display rig change times
    st.write("**Rig Change Times:**")
    if 'rig_change_times' in problem:
        # Create a heatmap of rig change times
        try:
            rig_change_times = problem['rig_change_times']
            fig = px.imshow(rig_change_times,
                           labels=dict(x="To Rig", y="From Rig", color="Change Time"),
                           x=[f"Rig {i+1}" for i in range(len(rig_change_times[0]))],
                           y=[f"Rig {i+1}" for i in range(len(rig_change_times))],
                           color_continuous_scale="Viridis")
            fig.update_layout(title="Rig Change Time Matrix")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.write("Rig change times:")
            st.json(problem['rig_change_times'])
            st.warning(f"Could not create heatmap: {str(e)}")
    else:
        st.warning("No rig change time information available")

    # Display solver settings
    st.write("**Solver Settings:**")
    if 'solver_settings' in problem:
        st.json(problem['solver_settings'])
    else:
        st.warning("No solver settings available")

# Function to visualize the solution as a Gantt chart
def visualize_solution(api_response):
    if api_response and 'api_response' in api_response and api_response['api_response']['status'] == 'success':
        solution = api_response['api_response']['solution']

        st.subheader("Solution Visualization")

        # Display makespan
        st.write(f"**Makespan:** {solution['objective_value']} time units")

        # Display job assignments
        st.write("**Job Assignments:**")
        assignments_df = pd.DataFrame([
            {"Job": job_name, "Assigned to Machine": machine_id} 
            for job_name, machine_id in solution['variables'].items()
        ])
        st.dataframe(assignments_df)

        # Create Gantt chart for scheduling problems
        if 'job_' in next(iter(solution['variables']), ''):
            st.subheader("Schedule Visualization (Gantt Chart)")

            # Organize data for Gantt chart
            jobs_by_machine = {}
            for job_name, machine_id in solution['variables'].items():
                machine_id = str(machine_id)
                if machine_id not in jobs_by_machine:
                    jobs_by_machine[machine_id] = []
                jobs_by_machine[machine_id].append(job_name)

            # Create Gantt chart data
            gantt_data = []
            start_time = datetime.now()

            for machine_id, jobs in jobs_by_machine.items():
                current_time = start_time
                for job in jobs:
                    # Assume each job takes 1 hour (can be adjusted based on actual data)
                    job_duration = timedelta(hours=1)
                    gantt_data.append(dict(
                        Task=f"Machine {machine_id}",
                        Start=current_time,
                        Finish=current_time + job_duration,
                        Resource=job
                    ))
                    current_time += job_duration

            if gantt_data:
                df = pd.DataFrame(gantt_data)
                fig = ff.create_gantt(df, colors=['#779ECB', '#AEC6CF', '#836953', '#CFCFC4'],
                                     index_col='Resource', show_colorbar=True, 
                                     group_tasks=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough data to create a Gantt chart")
    else:
        st.error("No valid solution to visualize")

# Problem Builder Page
if page == "Scheduling Problem Builder":
    st.title("Scheduling Problem Builder")

    with st.expander("About this tool", expanded=False):
        st.markdown("""
        This tool helps you build and solve scheduling problems using the OWPy library.

        You can:
        1. Define machines, jobs, and rig change times
        2. Visualize the scheduling problem
        3. Solve the problem to minimize makespan
        4. Visualize the solution as a Gantt chart

        For complex problems, you can also use the Chat Interface to describe your problem in natural language.
        """)

    # Initialize session state for scheduling problem components
    if 'machines' not in st.session_state:
        st.session_state.machines = [{'machine_id': 1, 'processing_time': 1}]
    if 'jobs' not in st.session_state:
        st.session_state.jobs = [{'job_id': 1, 'rig_id': 1}]
    if 'rig_change_times' not in st.session_state:
        st.session_state.rig_change_times = [[0, 1], [1, 0]]
    if 'solver_settings' not in st.session_state:
        st.session_state.solver_settings = {
            'max_time': 60,
            'use_heuristics': True,
            'solver_function': 'GLOBAL'
        }
    if 'solution' not in st.session_state:
        st.session_state.solution = None

    # Create columns for layout
    col1, col2 = st.columns([1, 1])

    # Problem Definition Section
    with col1:
        st.header("Scheduling Problem Definition")

        # Machines
        st.subheader("Machines")

        # Add new machine button
        if st.button("Add Machine"):
            st.session_state.machines.append({'machine_id': len(st.session_state.machines)+1, 'processing_time': 1})

        # Machine editor
        updated_machines = []
        for i, machine in enumerate(st.session_state.machines):
            st.markdown(f"**Machine {i+1}**")
            col_a, col_b, col_c = st.columns([2, 2, 1])

            with col_a:
                machine_id = st.number_input(f"Machine ID ##{i}", value=int(machine['machine_id']), min_value=1, step=1)

            with col_b:
                processing_time = st.number_input(f"Processing Time ##{i}", value=int(machine['processing_time']), min_value=1, step=1)

            with col_c:
                if st.button("Remove", key=f"remove_machine_{i}"):
                    continue

            updated_machines.append({
                'machine_id': machine_id,
                'processing_time': processing_time
            })

        st.session_state.machines = updated_machines

        # Jobs
        st.subheader("Jobs")

        # Add new job button
        if st.button("Add Job"):
            st.session_state.jobs.append({'job_id': len(st.session_state.jobs)+1, 'rig_id': 1})

        # Job editor
        updated_jobs = []
        for i, job in enumerate(st.session_state.jobs):
            st.markdown(f"**Job {i+1}**")
            col_a, col_b, col_c = st.columns([2, 2, 1])

            with col_a:
                job_id = st.number_input(f"Job ID ##{i}", value=int(job['job_id']), min_value=1, step=1)

            with col_b:
                rig_id = st.number_input(f"Rig ID ##{i}", value=int(job['rig_id']), min_value=1, step=1)

            with col_c:
                if st.button("Remove", key=f"remove_job_{i}"):
                    continue

            updated_jobs.append({
                'job_id': job_id,
                'rig_id': rig_id
            })

        st.session_state.jobs = updated_jobs

        # Rig Change Times
        st.subheader("Rig Change Times")

        # Determine the number of rigs
        all_rig_ids = set()
        for job in st.session_state.jobs:
            all_rig_ids.add(job['rig_id'])
        num_rigs = max(all_rig_ids) if all_rig_ids else 2

        # Ensure rig_change_times matrix is the right size
        current_size = len(st.session_state.rig_change_times)
        if current_size != num_rigs:
            # Resize the matrix
            new_matrix = []
            for i in range(num_rigs):
                row = []
                for j in range(num_rigs):
                    if i < current_size and j < current_size:
                        row.append(st.session_state.rig_change_times[i][j])
                    elif i == j:
                        row.append(0)  # No change time for same rig
                    else:
                        row.append(1)  # Default change time
                new_matrix.append(row)
            st.session_state.rig_change_times = new_matrix

        # Rig change time matrix editor
        st.write("Edit the rig change time matrix:")
        for i in range(num_rigs):
            cols = st.columns(num_rigs)
            for j in range(num_rigs):
                with cols[j]:
                    st.session_state.rig_change_times[i][j] = st.number_input(
                        f"From Rig {i+1} to Rig {j+1}",
                        value=st.session_state.rig_change_times[i][j],
                        min_value=0,
                        step=1,
                        key=f"rig_{i}_{j}"
                    )

        # Solver Settings
        st.subheader("Solver Settings")

        col_a, col_b = st.columns(2)

        with col_a:
            max_time = st.number_input(
                "Max Time (seconds)",
                value=st.session_state.solver_settings['max_time'],
                min_value=1,
                step=10
            )

        with col_b:
            use_heuristics = st.checkbox(
                "Use Heuristics",
                value=st.session_state.solver_settings['use_heuristics']
            )

        solver_function = st.selectbox(
            "Solver Function",
            options=['GLOBAL', 'TWO_STEP', 'SA', 'BranchAndBound'],
            index=['GLOBAL', 'TWO_STEP', 'SA', 'BranchAndBound'].index(
                st.session_state.solver_settings['solver_function']
            )
        )

        st.session_state.solver_settings = {
            'max_time': max_time,
            'use_heuristics': use_heuristics,
            'solver_function': solver_function
        }

        # Create the complete problem
        problem = {
            'machines': st.session_state.machines,
            'jobs': st.session_state.jobs,
            'rig_change_times': st.session_state.rig_change_times,
            'solver_settings': st.session_state.solver_settings
        }

        # Solve button
        if st.button("Solve Problem"):
            with st.spinner("Solving problem..."):
                # Call the API with the scheduling problem context and test mode setting
                api_response = call_chat_api("Solve this scheduling problem", context=problem, test_mode=test_mode)

                if api_response:
                    st.session_state.solution = api_response
                    st.success("Problem solved successfully!")
                else:
                    st.error("Failed to solve the problem. Check the API connection.")

    # Visualization Section
    with col2:
        # Problem visualization
        visualize_problem(problem)

        # Solution visualization (if available)
        if st.session_state.solution:
            visualize_solution(st.session_state.solution)

# Chat Interface Page
elif page == "Chat Interface":
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

# Footer
st.markdown("---")
st.markdown("OWPy Scheduling Assistant | Built with Streamlit")
