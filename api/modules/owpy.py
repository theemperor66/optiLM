import os
import requests
import random
from typing import Dict, Any
from datetime import datetime
from .models import SchedulingProblem, ScenarioOptimizeRequest, IOScenario, SolverSettings, SolverFunction, SolutionAndTime

# OWPy API URL and authentication
OWPY_API_URL = os.getenv("OWPY_API_URL", "https://api.optware.com/owpy")
OWPY_API_KEY = os.getenv("OWPY_API_KEY")  # API key for authentication

def convert_to_owpy_format(problem: SchedulingProblem) -> ScenarioOptimizeRequest:
    """
    Convert our internal SchedulingProblem format to the OWPy API format.

    This is a direct conversion since our SchedulingProblem model is already
    structured to match OWPy's expectations.

    Args:
        problem: The scheduling problem in our internal format

    Returns:
        ScenarioOptimizeRequest: The problem in OWPy API format
    """
    # Create the scenario from our scheduling problem
    scenario = IOScenario(
        Machines=problem.machines,
        Jobs=problem.jobs,
        Rigs=problem.rig_change_times
    )

    # Create solver settings
    settings = SolverSettings(
        MaxTime=problem.solver_settings.get("max_time", 60),
        UseHeuristics=problem.solver_settings.get("use_heuristics", True)
    )

    # Get solver function or default to GLOBAL
    solver_func_str = problem.solver_settings.get("solver_function", "GLOBAL")
    solver_func = SolverFunction(solver_func_str)

    # Create the request
    request = ScenarioOptimizeRequest(
        scenarioInput=scenario,
        solverFunc=solver_func,
        solver_settings=settings
    )

    return request

def generate_random_solution(problem: SchedulingProblem) -> Dict[str, Any]:
    """
    Generate a random solution for a scheduling problem.
    This is used in test mode to avoid calling the OWPy API.

    Args:
        problem: The scheduling problem to solve

    Returns:
        Dict[str, Any]: A randomly generated solution
    """
    # Get the number of jobs and machines
    num_jobs = len(problem.jobs)
    num_machines = len(problem.machines)

    # Randomly assign jobs to machines
    job_assignments = {}
    for i, job in enumerate(problem.jobs):
        machine_id = str(random.randint(1, num_machines))
        job_assignments[f"job_{i+1}"] = machine_id

    # Generate a random makespan between 5 and 20
    makespan = random.randint(5, 20)

    # Generate a random solve time between 0.1 and 2.0 seconds
    solve_time = round(random.uniform(0.1, 2.0), 1)

    # Return the solution in the same format as the OWPy API
    return {
        "status": "success",
        "solution": {
            "objective_value": makespan,
            "variables": job_assignments,
            "solve_time": solve_time
        }
    }

def call_owpy_api(problem: SchedulingProblem, test_mode: bool = False) -> Dict[str, Any]:
    """
    Call the OWPy API with the formulated scheduling problem or generate a random solution in test mode.

    Args:
        problem: The scheduling problem to solve
        test_mode: If True, generate a random solution instead of calling the OWPy API

    Returns:
        Dict[str, Any]: The API response or a randomly generated solution
    """
    # In test mode, generate a random solution
    if test_mode:
        return generate_random_solution(problem)
    try:
        # Convert our problem format to OWPy format
        owpy_request = convert_to_owpy_format(problem)

        # Set up headers with authentication
        headers = {
            "Content-Type": "application/json"
        }

        # Add API key if available
        if OWPY_API_KEY:
            headers["Authorization"] = f"Bearer {OWPY_API_KEY}"

        # Make the API call
        response = requests.post(
            f"{OWPY_API_URL}/api/v1/solve/",
            json=owpy_request.dict(),
            headers=headers
        )

        # Check for errors
        response.raise_for_status()

        # Parse the response
        solution_and_time = SolutionAndTime(**response.json())

        # Convert to our internal format
        return {
            "status": "success",
            "solution": {
                "objective_value": solution_and_time.makespan,  # Use makespan as objective value
                "variables": {
                    f"job_{i+1}": machine_id  # Map job to machine assignment
                    for machine_id, jobs in solution_and_time.machines_distribution.items()
                    for i, job_id in enumerate(jobs)
                },
                "solve_time": solution_and_time.time_needed_in_s
            }
        }
    except requests.RequestException as e:
        # Handle API errors
        error_msg = str(e)
        try:
            if hasattr(e, 'response') and e.response is not None:
                error_data = e.response.json()
                if 'detail' in error_data:
                    error_msg = error_data['detail']
        except:
            pass

        # Return error response
        return {
            "status": "error",
            "error": f"Error calling OWPy API: {error_msg}"
        }
    except Exception as e:
        # Handle other errors
        return {
            "status": "error",
            "error": f"Unexpected error: {str(e)}"
        }