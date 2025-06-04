from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import plotly.colors as pc
import uuid

def get_rig_id_mapping(problem: dict) -> tuple:
    """
    Extract all unique rig IDs from a problem and create a mapping to matrix indices.

    Args:
        problem: The scheduling problem dictionary

    Returns:
        tuple: (sorted_rig_ids, rig_id_to_index)
    """
    # Extract all unique rig IDs from jobs
    rig_ids = set()
    if problem.get("jobs"):
        for job in problem["jobs"]:
            if "rig_id" in job:
                rig_ids.add(job["rig_id"])

    # Also check machines for start_rig_id
    if problem.get("machines"):
        for machine in problem["machines"]:
            if "start_rig_id" in machine:
                rig_ids.add(machine["start_rig_id"])

    # If no rig IDs found, use sequential IDs based on matrix size
    if not rig_ids and problem.get("rig_change_times"):
        rig_ids = set(range(1, len(problem["rig_change_times"]) + 1))

    # Sort rig IDs for consistent display
    sorted_rig_ids = sorted(rig_ids)

    # Create a mapping from rig ID to index in the matrix
    rig_id_to_index = {rig_id: i for i, rig_id in enumerate(sorted_rig_ids)}

    return sorted_rig_ids, rig_id_to_index

def tooltip_text(term: str, explanation: str) -> str:
    """Create HTML for a term with a tooltip explanation."""
    return f"""
    <span style="text-decoration: underline dotted; cursor: help;" 
          title="{explanation}">
        {term}
    </span>
    """

# ---------------------------------------------------------------------------
# TODO: Future Enhancement - Improved Visualizations
# The visualizations could be enhanced in the following ways:
# 1. More intuitive graphics with better color coding and legends
# 2. Interactive elements that allow users to modify the problem by clicking
# 3. Animations to show how scheduling works
# 4. Alternative views of the same problem (timeline, resource allocation, etc.)
# 5. Progress indicators for problem completion

def visualize_problem(problem: dict):
    """Pretty-print a scheduling problem (machines, jobs, rigs, solver …)."""
    st.subheader("Scheduling Problem")

    # Machines ---------------------------------------------------------------
    st.markdown(f"**{tooltip_text('Machines', 'Resources that process jobs (e.g., ovens, printers, production lines)')}**", unsafe_allow_html=True)
    if problem.get("machines"):
        machines_df = pd.DataFrame(problem["machines"])

        # Check if any machine still has processing_time (deprecated)
        if any("processing_time" in m for m in problem["machines"]):
            st.warning("⚠️ Deprecated: 'processing_time' found on machines. "
                      "Processing time should now be specified on jobs.")

        st.dataframe(machines_df)
    else:
        st.warning("No machine information available.")

    # Jobs -------------------------------------------------------------------
    st.markdown(f"**{tooltip_text('Jobs', 'Tasks that need to be scheduled (e.g., baking a cake, printing a document)')}**", unsafe_allow_html=True)
    if problem.get("jobs"):
        st.dataframe(pd.DataFrame(problem["jobs"]))
    else:
        st.warning("No job information available.")

    # Rig-change matrix ------------------------------------------------------
    st.markdown(f"**{tooltip_text('Rig-change times', 'Time required to switch from one rig configuration to another')}**", unsafe_allow_html=True)
    if problem.get("rig_change_times"):
        try:
            # Get rig ID mapping
            sorted_rig_ids, rig_id_to_index = get_rig_id_mapping(problem)

            rct = problem["rig_change_times"]
            fig = px.imshow(
                rct,
                labels=dict(x="To rig", y="From rig", color="Δ time"),
                x=[f"Rig {rig_id}" for rig_id in sorted_rig_ids],
                y=[f"Rig {rig_id}" for rig_id in sorted_rig_ids],
                color_continuous_scale="Viridis",
            )
            fig.update_layout(title=f"Rig change-time matrix")
            st.plotly_chart(fig, use_container_width=True, key=f"rig_matrix_{str(uuid.uuid4())}")
        except Exception as e:  # noqa
            st.json(problem["rig_change_times"])
            st.warning(f"Could not create heatmap: {e}")
    else:
        st.warning("No rig change-time information available.")

    # Solver settings --------------------------------------------------------
    st.markdown(f"**{tooltip_text('Solver settings', 'Configuration parameters for the optimization algorithm')}**", unsafe_allow_html=True)
    st.json(problem.get("solver_settings") or {})

# ---------------------------------------------------------------------------
def visualize_solution(api_response: dict):
    """Render the solution returned by the OWPy API."""
    if not (api_response
            and api_response.get("api_response", {}).get("status") == "success"):
        st.error("No valid solution to visualise.")
        return

    solution = api_response["api_response"]["solution"]

    st.subheader("Solution")
    # Get timeunit from the problem data or use default
    timeunit = api_response.get("scheduling_problem", {}).get("timeunit", "minutes")
    st.markdown(f"**{tooltip_text('Makespan', 'The total time required to complete all jobs - a key metric in scheduling optimization')}:** {solution['objective_value']} {timeunit}", unsafe_allow_html=True)

    # -------- tabular assignment -------------------------------------------
    assign_df = (
        pd.DataFrame(
            [{"Job": j, "Machine": m} for j, m in solution["variables"].items()]
        )
        .sort_values("Machine")
        .reset_index(drop=True)
    )
    st.dataframe(assign_df)

    # -------- timeline / "Gantt" chart -------------------------------------
    st.markdown(f"<h3>{tooltip_text('Schedule', 'A visual representation of when each job is processed on each machine, showing the optimal sequence')}</h3>", unsafe_allow_html=True)

    # group jobs by machine
    gantt_rows = []
    t0 = datetime.now()

    # Get job durations and rig IDs from the problem if available
    job_durations = {}
    job_rig_ids = {}
    if api_response.get("scheduling_problem") and api_response["scheduling_problem"].get("jobs"):
        for job in api_response["scheduling_problem"]["jobs"]:
            job_id = f"job_{job['job_id']}"
            job_durations[job_id] = job.get("processing_time", 1)
            job_rig_ids[job_id] = job.get("rig_id", 1)

    # Get rig change times from the problem if available
    rig_change_times = api_response.get("scheduling_problem", {}).get("rig_change_times", [])

    # Track machine start times and last rig used
    machine_start_times = {}
    machine_last_rig = {}

    # Sort jobs by machine for better visualization
    sorted_jobs = sorted(solution["variables"].items(), key=lambda x: int(x[1]))

    for job, machine in sorted_jobs:
        # Convert machine to int to ensure arithmetic operations work
        machine_int = int(machine)

        # Get job duration (default to 1 if not found)
        job_duration = job_durations.get(job, 1)

        # Get timeunit from the problem data or use default
        timeunit = api_response.get("scheduling_problem", {}).get("timeunit", "minutes")

        # Convert duration to timedelta based on timeunit
        if timeunit == "hours":
            duration = timedelta(hours=job_duration)
            time_label = "h"
        elif timeunit == "seconds":
            duration = timedelta(seconds=job_duration)
            time_label = "s"
        else:  # default to minutes
            duration = timedelta(minutes=job_duration)
            time_label = "min"

        # Get the current job's rig ID
        current_rig_id = job_rig_ids.get(job, 1)

        # Calculate start time based on machine's last job end time
        if machine not in machine_start_times:
            # Use the same timeunit for the initial offset
            if timeunit == "hours":
                machine_start_times[machine] = t0 + (machine_int - 1) * timedelta(hours=1)
            elif timeunit == "seconds":
                machine_start_times[machine] = t0 + (machine_int - 1) * timedelta(seconds=1)
            else:  # default to minutes
                machine_start_times[machine] = t0 + (machine_int - 1) * timedelta(minutes=1)

            # Initialize last rig for this machine
            machine_last_rig[machine] = current_rig_id
        else:
            # Check if there's a rig change needed
            last_rig_id = machine_last_rig.get(machine, 1)

            if last_rig_id != current_rig_id and rig_change_times:
                # Get rig change time if available
                try:
                    # Get rig ID mapping from the scheduling problem
                    if api_response.get("scheduling_problem"):
                        sorted_rig_ids, rig_id_to_index = get_rig_id_mapping(api_response["scheduling_problem"])
                    else:
                        # Create a default mapping based on the rig change times matrix
                        rig_ids = set(range(1, len(rig_change_times) + 1))
                        sorted_rig_ids = sorted(rig_ids)
                        rig_id_to_index = {rig_id: i for i, rig_id in enumerate(sorted_rig_ids)}

                    # Get indices using the mapping
                    from_idx = rig_id_to_index.get(last_rig_id, 0)
                    to_idx = rig_id_to_index.get(current_rig_id, 0)

                    if 0 <= from_idx < len(rig_change_times) and 0 <= to_idx < len(rig_change_times[0]):
                        rig_change_time = rig_change_times[from_idx][to_idx]

                        # Create a timedelta for the rig change time
                        if timeunit == "hours":
                            rig_change_duration = timedelta(hours=rig_change_time)
                        elif timeunit == "seconds":
                            rig_change_duration = timedelta(seconds=rig_change_time)
                        else:  # default to minutes
                            rig_change_duration = timedelta(minutes=rig_change_time)

                        # Add rig change to Gantt chart
                        rig_change_start = machine_start_times[machine]
                        rig_change_finish = rig_change_start + rig_change_duration

                        gantt_rows.append(dict(
                            Machine=f"Machine {machine}",
                            Job=f"Rig change {last_rig_id}→{current_rig_id} ({rig_change_time}{time_label})",
                            Start=rig_change_start,
                            Finish=rig_change_finish,
                            Type="RigChange"  # Mark as a rig change for coloring
                        ))

                        # Update machine start time to after the rig change
                        machine_start_times[machine] = rig_change_finish
                except (IndexError, TypeError) as e:
                    # If there's an error accessing rig change times, log it but continue
                    print(f"Error calculating rig change time: {e}")

            # Update the last rig used for this machine
            machine_last_rig[machine] = current_rig_id

        start = machine_start_times[machine]
        finish = start + duration

        # Update machine's next job start time
        machine_start_times[machine] = finish

        gantt_rows.append(dict(
            Machine=f"Machine {machine}",
            Job=f"{job} ({job_duration}{time_label})", 
            Start=start, 
            Finish=finish,
            Type="Job"  # Mark as a job for coloring
        ))

    df = pd.DataFrame(gantt_rows)
    if df.empty:
        st.warning("Nothing to plot.")
        return

    # Use Type for coloring (Job vs RigChange)
    # Create a custom color map with different colors for jobs and rig changes
    if "Type" in df.columns:
        # Define colors for jobs and rig changes
        color_map = {
            "Job": "#636EFA",  # Blue for jobs
            "RigChange": "#EF553B"  # Red for rig changes
        }

        fig = px.timeline(
            df, x_start="Start", x_end="Finish", y="Machine",
            color="Type", color_discrete_map=color_map, hover_name="Job"
        )
    else:
        # Fallback to original coloring if Type column is not present
        res_uniques = df["Job"].unique()
        palette = pc.qualitative.Plotly * ((len(res_uniques) // 10) + 1)
        color_map = {res: palette[i] for i, res in enumerate(res_uniques)}

        fig = px.timeline(
            df, x_start="Start", x_end="Finish", y="Machine",
            color="Job", color_discrete_map=color_map, hover_name="Job"
        )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True, key=f"gantt_chart_{str(uuid.uuid4())}")
