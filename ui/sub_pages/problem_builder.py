# (imports unchanged)
import json
import streamlit as st
from modules.api_client import call_chat_api, get_available_solvers
from modules.visualization import visualize_problem, visualize_solution
from modules.problem_io import (
    get_default_problem,
    to_example_format,
    from_example_format,
)

def show_problem_builder(test_mode: bool = False):
    st.title("Scheduling Problem Builder")

    # ---------------- Initialise state -------------------------------------
    ss = st.session_state  # shorthand

    ss.setdefault(
        "current_problem",
        {
            "machines": [dict(machine_id=1, start_rig_id=1)],
            "jobs": [dict(job_id=1, rig_id=1, processing_time=1)],
            "rig_change_times": [[0, 1], [1, 0]],
            "solver_settings": dict(max_time=60, use_heuristics=True, solver_function="GLOBAL"),
        },
    )
    ss.setdefault("current_solution", None)

    # Sidebar utilities ----------------------------------------------------
    with st.sidebar:
        st.markdown("### Problem File")
        if st.button("🗑 Reset problem", key="reset_builder"):
            ss.current_problem = get_default_problem()
            ss.current_solution = None
            ss.builder_sync_key = ""
            st.rerun()

        st.download_button(
            label="⬇️ Download problem", 
            data=json.dumps(to_example_format(ss.current_problem), indent=2),
            file_name="problem.json",
            mime="application/json",
        )

        uploaded = st.file_uploader("Load problem", type="json", key="problem_uploader_" + str(hash("problem_uploader")))
        if uploaded is not None:
            # Only process the file if we haven't done so for this file already
            file_id = hash(uploaded.name + str(uploaded.size))
            if ss.get("last_uploaded_file_id") != file_id:
                try:
                    # Immediately read the file content before it potentially gets lost
                    content = uploaded.read()
                    uploaded.seek(0)  # Reset pointer
                    
                    # Create a backup of the file content in session state
                    ss[f"file_backup_{file_id}"] = content
                    
                    try:
                        # Try to parse the file
                        data = json.loads(content.decode("utf-8"))
                        ss.current_problem = from_example_format(data)
                        ss.current_solution = None
                        ss.builder_sync_key = ""
                        ss.last_uploaded_file_id = file_id
                        st.success("Problem loaded")
                        st.rerun()
                    except json.JSONDecodeError as je:
                        st.error(f"Invalid JSON format: {je}")
                except Exception as e:
                    # If we have a backup of the file content, try to use that
                    if f"file_backup_{file_id}" in ss:
                        try:
                            backup_content = ss[f"file_backup_{file_id}"]
                            data = json.loads(backup_content.decode("utf-8"))
                            ss.current_problem = from_example_format(data)
                            ss.current_solution = None
                            ss.builder_sync_key = ""
                            ss.last_uploaded_file_id = file_id
                            st.success("Problem loaded from backup")
                            st.rerun()
                        except Exception as be:
                            st.error(f"Failed to load file: {str(e)}\nBackup attempt also failed: {str(be)}")
                    else:
                        st.error(f"Failed to load file: {str(e)}")
            elif ss.get("last_upload_success", False):
                st.success("Problem previously loaded")
            else:
                st.warning("Upload was processed previously but failed. Please try a different file.")
                
        # Store upload success status for reference in subsequent reruns
        if uploaded is not None and ss.get("last_uploaded_file_id") == hash(uploaded.name + str(uploaded.size)):
            ss["last_upload_success"] = True

    # Sync builder-specific state with the shared problem
    prob_hash = str(ss.current_problem)
    if ss.get("builder_sync_key") != prob_hash:
        # Process machines - ensure all IDs are at least 1
        machines = ss.current_problem.get("machines", [dict(machine_id=1, start_rig_id=1)])
        normalized_machines = []
        for machine in machines:
            normalized_machines.append({
                "machine_id": max(1, int(machine.get("machine_id", 1))),
                "start_rig_id": max(1, int(machine.get("start_rig_id", 1)))
            })
        ss.machines = normalized_machines
        
        # Process jobs - ensure all IDs are at least 1
        jobs = ss.current_problem.get("jobs", [dict(job_id=1, rig_id=1, processing_time=1)])
        normalized_jobs = []
        for job in jobs:
            normalized_jobs.append({
                "job_id": max(1, int(job.get("job_id", 1))),
                "rig_id": max(1, int(job.get("rig_id", 1))),
                "processing_time": max(1, int(job.get("processing_time", 1)))
            })
        ss.jobs = normalized_jobs
        
        ss.rig_change_times = ss.current_problem.get("rig_change_times", [[0, 1], [1, 0]])
        ss.solver_settings = ss.current_problem.get(
            "solver_settings", dict(max_time=60, use_heuristics=True, solver_function="GLOBAL")
        )
        ss.solution = ss.current_solution
        ss.builder_sync_key = prob_hash

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
        problem = dict(
            machines=ss.machines,
            jobs=ss.jobs,
            rig_change_times=ss.rig_change_times,
            solver_settings=ss.solver_settings,
        )
        ss.current_problem = problem

        if st.button("🚀 Solve problem"):
            with st.spinner("Contacting solver…"):
                reply = call_chat_api("Solve this scheduling problem",
                                      context=problem, test_mode=test_mode)
            if reply:
                ss.solution = reply
                ss.current_solution = reply
                st.success("Solved!")
            else:
                st.error("Solver failed – see sidebar logs.")

    # -----------------------------------------------------------------------
    with col2:
        visualize_problem(ss.current_problem)
        if ss.current_solution:
            visualize_solution(ss.current_solution)
