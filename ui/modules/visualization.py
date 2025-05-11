from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import plotly.colors as pc
import uuid

# ---------------------------------------------------------------------------
def visualize_problem(problem: dict):
    """Pretty-print a scheduling problem (machines, jobs, rigs, solver …)."""
    st.subheader("Scheduling Problem")

    # Machines ---------------------------------------------------------------
    st.write("**Machines**")
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
    st.write("**Jobs**")
    if problem.get("jobs"):
        st.dataframe(pd.DataFrame(problem["jobs"]))
    else:
        st.warning("No job information available.")

    # Rig-change matrix ------------------------------------------------------
    st.write("**Rig-change times**")
    if problem.get("rig_change_times"):
        try:
            rct = problem["rig_change_times"]
            fig = px.imshow(
                rct,
                labels=dict(x="To rig", y="From rig", color="Δ time"),
                x=[f"Rig {i+1}" for i in range(len(rct[0]))],
                y=[f"Rig {i+1}" for i in range(len(rct))],
                color_continuous_scale="Viridis",
            )
            fig.update_layout(title="Rig change-time matrix")
            st.plotly_chart(fig, use_container_width=True, key=f"rig_matrix_{str(uuid.uuid4())}")
        except Exception as e:  # noqa
            st.json(problem["rig_change_times"])
            st.warning(f"Could not create heatmap: {e}")
    else:
        st.warning("No rig change-time information available.")

    # Solver settings --------------------------------------------------------
    st.write("**Solver settings**")
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
    st.write(f"**Makespan:** {solution['objective_value']} time units")

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
    st.subheader("Schedule")

    # group jobs by machine
    gantt_rows = []
    t0 = datetime.now()

    # Get job durations from the problem if available
    job_durations = {}
    if api_response.get("scheduling_problem") and api_response["scheduling_problem"].get("jobs"):
        for job in api_response["scheduling_problem"]["jobs"]:
            job_id = f"job_{job['job_id']}"
            job_durations[job_id] = job.get("processing_time", 1)

    # Track machine start times
    machine_start_times = {}

    # Sort jobs by machine for better visualization
    sorted_jobs = sorted(solution["variables"].items(), key=lambda x: int(x[1]))

    for job, machine in sorted_jobs:
        # Convert machine to int to ensure arithmetic operations work
        machine_int = int(machine)

        # Get job duration (default to 1 if not found)
        job_duration = job_durations.get(job, 1)
        duration = timedelta(hours=job_duration)

        # Calculate start time based on machine's last job end time
        if machine not in machine_start_times:
            machine_start_times[machine] = t0 + (machine_int - 1) * timedelta(hours=1)

        start = machine_start_times[machine]
        finish = start + duration

        # Update machine's next job start time
        machine_start_times[machine] = finish

        gantt_rows.append(dict(Machine=f"Machine {machine}",
                               Job=f"{job} ({job_duration}h)", 
                               Start=start, Finish=finish))

    df = pd.DataFrame(gantt_rows)
    if df.empty:
        st.warning("Nothing to plot.")
        return

    # colour palette that scales
    res_uniques = df["Job"].unique()
    palette = pc.qualitative.Plotly * ((len(res_uniques) // 10) + 1)
    color_map = {res: palette[i] for i, res in enumerate(res_uniques)}

    fig = px.timeline(
        df, x_start="Start", x_end="Finish", y="Machine",
        color="Job", color_discrete_map=color_map, hover_name="Job"
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True, key=f"gantt_chart_{str(uuid.uuid4())}")
