import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
from datetime import datetime, timedelta

def visualize_problem(problem):
    """
    Visualize a scheduling problem.
    
    Args:
        problem (dict): The scheduling problem to visualize
    """
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

def visualize_solution(api_response):
    """
    Visualize a solution as a Gantt chart.
    
    Args:
        api_response (dict): The API response containing the solution
    """
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