from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import plotly.colors as pc

# ---------------------------------------------------------------------------
def visualize_problem(problem: dict):
    """Pretty-print a scheduling problem (machines, jobs, rigs, solver …)."""
    st.subheader("Scheduling Problem")

    # Machines ---------------------------------------------------------------
    st.write("**Machines**")
    if problem.get("machines"):
        st.dataframe(pd.DataFrame(problem["machines"]))
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
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:  # noqa
            st.json(problem["rig_change_times"])
            st.warning(f"Could not create heatmap: {e}")
    else:
        st.warning("No rig change-time information available.")

    # Solver settings --------------------------------------------------------
    st.write("**Solver settings**")
    st.json(problem.get("solver_settings", "—"))

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
    per_job_duration = timedelta(hours=1)           # TODO: replace with real data
    for job, machine in solution["variables"].items():
        # Convert machine to int to ensure arithmetic operations work
        machine_int = int(machine)
        start = t0 + (machine_int - 1) * per_job_duration
        finish = start + per_job_duration
        gantt_rows.append(dict(Machine=f"Machine {machine}",
                               Job=job, Start=start, Finish=finish))

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
    st.plotly_chart(fig, use_container_width=True)
