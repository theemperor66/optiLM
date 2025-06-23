import json
from typing import Dict, Any

DEFAULT_PROBLEM: Dict[str, Any] = {
    "machines": [dict(machine_id=1, start_rig_id=1)],
    "jobs": [dict(job_id=1, rig_id=1, processing_time=1)],
    "rig_change_times": [[0, 1], [1, 0]],
    "solver_settings": dict(max_time=60, use_heuristics=True, solver_function="GLOBAL"),
}


def get_default_problem() -> Dict[str, Any]:
    """Return a new copy of the default scheduling problem."""
    return json.loads(json.dumps(DEFAULT_PROBLEM))


def to_example_format(problem: Dict[str, Any]) -> Dict[str, Any]:
    """Convert internal problem representation to the example JSON format."""
    jobs = [
        {"Duration": j.get("processing_time", 1), "Rig": j.get("rig_id", 1) - 1}
        for j in problem.get("jobs", [])
    ]
    machines = [
        {"Rig": m.get("start_rig_id", 1) - 1}
        for m in problem.get("machines", [])
    ]
    return {
        "Jobs": jobs,
        "Machines": machines,
        "Rigs": problem.get("rig_change_times", []),
    }


def from_example_format(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a problem in the example format to the internal representation."""
    jobs = [
        {
            "job_id": i + 1,
            "rig_id": j.get("Rig", 0) + 1,
            "processing_time": j.get("Duration", 1),
        }
        for i, j in enumerate(data.get("Jobs", []))
    ]
    machines = [
        {
            "machine_id": i + 1,
            "start_rig_id": m.get("Rig", 0) + 1,
        }
        for i, m in enumerate(data.get("Machines", []))
    ]
    rig_change_times = data.get("Rigs", [])

    problem = get_default_problem()
    if machines:
        problem["machines"] = machines
    if jobs:
        problem["jobs"] = jobs
    if rig_change_times:
        problem["rig_change_times"] = rig_change_times
    return problem
