import os
import requests
import random
import logging
import time
import traceback
from typing import Dict, Any, Optional
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .models import SchedulingProblem, ScenarioOptimizeRequest, IOScenario, SolverSettings, SolverFunction, SolutionAndTime

# Configure logging
logger = logging.getLogger("api.owpy")

# OWPy API URL and authentication
OWPY_API_URL = os.getenv("OWPY_API_URL", "https://api.optware.com/owpy")
OWPY_AUTH_CREDENTIALS = os.getenv("OWPY_AUTH_CREDENTIALS")  # Authentication credentials

if OWPY_API_URL:
    logger.info(f"OWPy API URL configured: {OWPY_API_URL}")
else:
    logger.warning("OWPy API URL not configured, using default")

if OWPY_AUTH_CREDENTIALS:
    logger.info("OWPy authentication credentials configured")
else:
    logger.warning("OWPy authentication credentials not configured")

def convert_to_owpy_format(problem: SchedulingProblem) -> ScenarioOptimizeRequest:
    """
    Convert our internal SchedulingProblem format to the OWPy API format.

    This is a direct conversion since our SchedulingProblem model is already
    structured to match OWPy's expectations.

    Args:
        problem: The scheduling problem in our internal format

    Returns:
        ScenarioOptimizeRequest: The problem in OWPy API format

    Raises:
        ValueError: If the problem format is invalid
    """
    logger.debug("Converting scheduling problem to OWPy format")
    start_time = time.time()

    try:
        # Validate problem structure
        if not problem.machines:
            logger.error("Invalid problem: no machines defined")
            raise ValueError("No machines defined in the scheduling problem")

        if not problem.jobs:
            logger.error("Invalid problem: no jobs defined")
            raise ValueError("No jobs defined in the scheduling problem")

        if not problem.rig_change_times:
            logger.error("Invalid problem: no rig change times defined")
            raise ValueError("No rig change times defined in the scheduling problem")

        # Create the scenario from our scheduling problem
        # Clone jobs so we don't mutate original and convert to dict format
        owpy_jobs = [j.model_dump() for j in problem.jobs]

        scenario = IOScenario(
            Machines=problem.machines,
            Jobs=owpy_jobs,
            Rigs=problem.rig_change_times
        )

        # Create solver settings
        settings = SolverSettings(
            MaxTime=problem.solver_settings.get("max_time", 60),
            UseHeuristics=problem.solver_settings.get("use_heuristics", True)
        )

        # Get solver function or default to GLOBAL
        solver_func_str = problem.solver_settings.get("solver_function", "GLOBAL")
        logger.debug(f"Using solver function: {solver_func_str}")
        solver_func = SolverFunction(solver_func_str)

        # Create the request
        request = ScenarioOptimizeRequest(
            scenarioInput=scenario,
            solverFunc=solver_func,
            solver_settings=settings
        )

        process_time = time.time() - start_time
        logger.debug(f"Converted problem to OWPy format in {process_time:.3f}s")

        return request

    except Exception as e:
        logger.error(f"Error converting problem to OWPy format: {str(e)}")
        logger.error(traceback.format_exc())
        raise ValueError(f"Failed to convert problem to OWPy format: {str(e)}")

def generate_random_solution(problem: SchedulingProblem) -> Dict[str, Any]:
    """
    Generate a random solution for a scheduling problem.
    This is used in test mode to avoid calling the OWPy API.

    Args:
        problem: The scheduling problem to solve

    Returns:
        Dict[str, Any]: A randomly generated solution
    """
    logger.info("Generating random solution for test mode")
    start_time = time.time()

    try:
        # Validate problem structure
        if not problem.machines:
            logger.error("Invalid problem: no machines defined")
            return {
                "status": "error",
                "error": "No machines defined in the scheduling problem"
            }

        if not problem.jobs:
            logger.error("Invalid problem: no jobs defined")
            return {
                "status": "error",
                "error": "No jobs defined in the scheduling problem"
            }

        # Get the number of jobs and machines
        num_jobs = len(problem.jobs)
        num_machines = len(problem.machines)
        logger.debug(f"Generating random solution for {num_jobs} jobs and {num_machines} machines")

        # Randomly assign jobs to machines
        job_assignments = {}
        for i, job in enumerate(problem.jobs):
            machine_id = str(random.randint(1, num_machines))
            job_assignments[f"job_{i+1}"] = machine_id

        # Generate a random makespan between 5 and 20
        makespan = random.randint(5, 20)

        # Generate a random solve time between 0.1 and 2.0 seconds
        solve_time = round(random.uniform(0.1, 2.0), 1)

        # Create the solution in the same format as the OWPy API
        solution = {
            "status": "success",
            "solution": {
                "objective_value": makespan,
                "variables": job_assignments,
                "solve_time": solve_time
            }
        }

        process_time = time.time() - start_time
        logger.info(f"Random solution generated in {process_time:.3f}s")
        logger.debug(f"Random solution: {solution}")

        return solution

    except Exception as e:
        logger.error(f"Error generating random solution: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "error": f"Failed to generate random solution: {str(e)}"
        }

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout))
)
def call_owpy_api(problem: SchedulingProblem, test_mode: bool = False) -> Dict[str, Any]:
    """
    Call the OWPy API with the formulated scheduling problem or generate a random solution in test mode.

    This function includes retry logic for handling transient network errors.

    Args:
        problem: The scheduling problem to solve
        test_mode: If True, generate a random solution instead of calling the OWPy API

    Returns:
        Dict[str, Any]: The API response or a randomly generated solution
    """
    request_id = f"owpy-{int(time.time())}"
    logger.info(f"OWPy API request {request_id} initiated (test_mode={test_mode})")
    start_time = time.time()

    # In test mode, generate a random solution
    if test_mode:
        logger.info(f"Test mode enabled for request {request_id}, using random solution")
        return generate_random_solution(problem)

    # Check if OWPy authentication credentials are set
    if not OWPY_AUTH_CREDENTIALS:
        logger.error(f"OWPy authentication credentials not configured for request {request_id}")
        return {
            "status": "error",
            "error": "OWPy authentication credentials not configured. Please set the OWPY_AUTH_CREDENTIALS environment variable."
        }

    try:
        # Convert our problem format to OWPy format
        logger.debug(f"Converting problem to OWPy format for request {request_id}")
        owpy_request = convert_to_owpy_format(problem)

        # Set up headers with authentication
        headers = {
            "Content-Type": "application/json"
        }

        # Add authentication credentials if available
        if OWPY_AUTH_CREDENTIALS:
            headers["Authorization"] = f"Bearer {OWPY_AUTH_CREDENTIALS}"
            logger.debug(f"Added authentication header for request {request_id}")

        # Prepare request data
        api_url = f"{OWPY_API_URL}/api/v1/solve/"
        request_data = owpy_request.dict()
        logger.debug(f"Sending request {request_id} to {api_url}")

        # Make the API call
        logger.info(f"Making OWPy API call for request {request_id}")
        response = requests.post(
            api_url,
            json=request_data,
            headers=headers,
            timeout=120  # Add timeout to prevent hanging requests
        )

        # Check for errors
        response.raise_for_status()
        logger.debug(f"Received response for request {request_id}: status_code={response.status_code}")

        # Parse the response
        try:
            logger.debug(f"Parsing JSON response for request {request_id}")
            response_data = response.json()
            solution_and_time = SolutionAndTime(**response_data)

            # Validate response structure
            if not hasattr(solution_and_time, 'makespan') or not hasattr(solution_and_time, 'machines_distribution'):
                logger.error(f"Invalid response structure for request {request_id}")
                return {
                    "status": "error",
                    "error": "Invalid response from OWPy API: missing required fields"
                }

            # Convert to our internal format
            result = {
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

            process_time = time.time() - start_time
            logger.info(f"OWPy API request {request_id} completed successfully in {process_time:.3f}s")
            logger.debug(f"OWPy API response for request {request_id}: {result}")

            return result

        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"Error parsing OWPy API response for request {request_id}: {str(e)}")
            logger.error(f"Response content: {response.text[:500]}...")
            return {
                "status": "error",
                "error": f"Invalid response format from OWPy API: {str(e)}"
            }

    except requests.RequestException as e:
        # Handle API errors
        process_time = time.time() - start_time
        logger.error(f"OWPy API request error for {request_id} after {process_time:.3f}s: {str(e)}")

        error_msg = str(e)
        status_code = None

        try:
            if hasattr(e, 'response') and e.response is not None:
                status_code = e.response.status_code
                logger.debug(f"Error status code for request {request_id}: {status_code}")

                try:
                    error_data = e.response.json()
                    if 'detail' in error_data:
                        error_msg = error_data['detail']
                        logger.debug(f"Error detail for request {request_id}: {error_msg}")
                except (ValueError, TypeError) as json_err:
                    logger.warning(f"Could not parse error response JSON for request {request_id}: {str(json_err)}")
                    logger.debug(f"Error response content: {e.response.text[:500]}...")
        except Exception as parse_err:
            logger.warning(f"Error extracting details from exception for request {request_id}: {str(parse_err)}")

        # Return error response with status code if available
        error_response = {
            "status": "error",
            "error": f"Error calling OWPy API: {error_msg}"
        }

        if status_code:
            error_response["status_code"] = status_code

        return error_response

    except Exception as e:
        # Handle other errors
        process_time = time.time() - start_time
        logger.error(f"Unexpected error for OWPy API request {request_id} after {process_time:.3f}s: {str(e)}")
        logger.error(traceback.format_exc())

        return {
            "status": "error",
            "error": f"Unexpected error processing OWPy API request: {str(e)}"
        }
