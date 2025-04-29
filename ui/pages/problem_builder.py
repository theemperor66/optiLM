import streamlit as st
from modules.api_client import call_chat_api
from modules.visualization import visualize_problem, visualize_solution

def show_problem_builder(test_mode=False):
    """
    Display the Problem Builder page.

    Args:
        test_mode (bool, optional): Whether to use test mode. Defaults to False.
    """
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
