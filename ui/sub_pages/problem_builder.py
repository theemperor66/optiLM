# (imports unchanged)
import streamlit as st
from modules.api_client import call_chat_api, get_available_solvers
from modules.visualization import visualize_problem, visualize_solution

def show_problem_builder(test_mode: bool = False):
    st.title("Scheduling Problem Builder")

    # ---------------- Initialise state -------------------------------------
    ss = st.session_state                      # shorthand

    ss.setdefault("machines",       [dict(machine_id=1, start_rig_id=1)])
    ss.setdefault("jobs",           [dict(job_id=1, rig_id=1, processing_time=1)])
    ss.setdefault("rig_change_times", [[0, 1], [1, 0]])
    ss.setdefault("solver_settings", dict(max_time=60,
                                          use_heuristics=True,
                                          solver_function="GLOBAL"))
    ss.setdefault("solution", None)

    col1, col2 = st.columns([1, 1])

    # -----------------------------------------------------------------------
    with col1:
        st.header("Define problem")

        # ---------- MACHINES ----------------------------------------------
        st.subheader("Machines")

        # ------ header row -------------------------------------------------------
        mh1, mh2, mh3 = st.columns([2, 2, 1])
        mh1.markdown("**Machine ID**")
        mh2.markdown("**Start rig**")
        mh3.markdown(" ")

        if st.button("➕ Add machine"):
            ss.machines.append(dict(machine_id=len(ss.machines)+1, start_rig_id=1))

        new_machines = []
        for i, m in enumerate(ss.machines):
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                mid = st.number_input(
                    "Machine ID",
                    key=f"m_id_{i}",
                    value=int(m["machine_id"]),
                    min_value=1,
                    step=1,
                    help="Unique integer identifier")
            with c2:
                srig = st.number_input(
                    "Start rig",
                    key=f"m_srig_{i}",
                    value=int(m.get("start_rig_id", 1)),
                    min_value=1,
                    step=1,
                    help="Rig that is mounted on this machine **before** scheduling starts")
            with c3:
                if st.button("🗑", key=f"rm_m_{i}"):
                    continue
            new_machines.append(dict(machine_id=mid, start_rig_id=srig))
        ss.machines = new_machines

        # ---------- JOBS ---------------------------------------------------
        st.subheader("Jobs")

        # ------- header row -----------------------------------------------------
        jh1, jh2, jh3, jh4 = st.columns([2, 2, 2, 1])
        jh1.markdown("**Job ID**")
        jh2.markdown("**Req. rig**")
        jh3.markdown("**Proc. time**")
        jh4.markdown(" ")

        if st.button("➕ Add job"):
            ss.jobs.append(dict(job_id=len(ss.jobs) + 1, rig_id=1, processing_time=1))

        new_jobs = []
        for i, j in enumerate(ss.jobs):
            c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
            with c1:
                jid = st.number_input("Job ID",
                                      key=f"j_id_{i}", value=int(j["job_id"]),
                                      min_value=1, step=1,
                                      help="Unique integer identifier")
            with c2:
                rid = st.number_input("Rig ID",
                                      key=f"j_rig_{i}", value=int(j["rig_id"]),
                                      min_value=1, step=1,
                                      help="Rig required for this job")
            with c3:
                ptime = st.number_input("Proc. time",
                                        key=f"j_pt_{i}", 
                                        value=int(j.get("processing_time", 1)),
                                        min_value=1, step=1,
                                        help="Processing time in chosen time units")
            with c4:
                if st.button("🗑", key=f"rm_j_{i}"):
                    continue
            new_jobs.append(dict(job_id=jid, rig_id=rid, processing_time=ptime))
        ss.jobs = new_jobs

        # ---------- RIG CHANGE MATRIX -------------------------------------
        st.subheader("Rig change-times")
        rig_ids = {j["rig_id"] for j in ss.jobs}
        n_rigs  = max(rig_ids) if rig_ids else 2

        # keep matrix square & sized
        rct = [[ss.rig_change_times[i][j] if i < len(ss.rig_change_times)
                                         and j < len(ss.rig_change_times)
                else (0 if i == j else 1)
                for j in range(n_rigs)]
                for i in range(n_rigs)]

        edit_matrix = []
        st.write("Edit matrix (rows = from, columns = to):")
        for i in range(n_rigs):
            cols = st.columns(n_rigs)
            edit_row = []
            for j in range(n_rigs):
                with cols[j]:
                    edit_row.append(
                        st.number_input(
                            label=f"{i}->{j}", key=f"rct_{i}_{j}",
                            value=int(rct[i][j]), min_value=0, step=1,
                            label_visibility="collapsed",
                            help=f"Change time from rig {i+1} to {j+1}",
                        )
                    )
            edit_matrix.append(edit_row)
        ss.rig_change_times = edit_matrix   # stored only once

        # ---------- SOLVER SETTINGS ----------------------------------------
        st.subheader("Solver settings")
        cs1, cs2, cs3 = st.columns(3)
        with cs1:
            mt = st.number_input("Max time (s)", min_value=1,
                                 value=int(ss.solver_settings["max_time"]))
        with cs2:
            heur = st.checkbox("Use heuristics",
                               value=ss.solver_settings["use_heuristics"])
        with cs3:
            # Get available solvers from API or use cached result
            available_solvers = get_available_solvers()

            # Find the index of the current solver in the available solvers list
            # Default to 0 (first solver) if not found
            current_solver = ss.solver_settings["solver_function"]
            try:
                solver_index = available_solvers.index(current_solver)
            except ValueError:
                solver_index = 0

            func = st.selectbox("Function",
                                available_solvers,
                                index=solver_index)
        ss.solver_settings = dict(max_time=mt,
                                  use_heuristics=heur,
                                  solver_function=func)

        # ---------- SOLVE ---------------------------------------------------
        problem = dict(machines=ss.machines,
                       jobs=ss.jobs,
                       rig_change_times=ss.rig_change_times,
                       solver_settings=ss.solver_settings)

        if st.button("🚀 Solve problem"):
            with st.spinner("Contacting solver…"):
                reply = call_chat_api("Solve this scheduling problem",
                                      context=problem, test_mode=test_mode)
            if reply:
                ss.solution = reply
                st.success("Solved!")
            else:
                st.error("Solver failed – see sidebar logs.")

    # -----------------------------------------------------------------------
    with col2:
        visualize_problem(dict(machines=ss.machines, jobs=ss.jobs,
                               rig_change_times=ss.rig_change_times,
                               solver_settings=ss.solver_settings))
        if ss.solution:
            visualize_solution(ss.solution)
