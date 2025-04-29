from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
import requests
import json
import os
from typing import Dict, Any, Optional, List, Union
import google.generativeai as genai
from dotenv import load_dotenv
from enum import Enum

# Load environment variables from .env file
load_dotenv()

# Configure Google Generative AI with API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    print("Warning: GOOGLE_API_KEY not found in environment variables")

app = FastAPI(
    title="OWPy Chatbot API",
    description="A chatbot API that helps users with OWPy optimization library",
    version="1.0.0"
)

# OWPy API URL and authentication
OWPY_API_URL = os.getenv("OWPY_API_URL", "https://api.optware.com/owpy")
OWPY_API_KEY = os.getenv("OWPY_API_KEY")  # API key for authentication

class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None
    test_mode: bool = False

class ChatResponse(BaseModel):
    response: str
    requires_support: bool = False
    scheduling_problem: Optional[Dict[str, Any]] = None
    api_response: Optional[Dict[str, Any]] = None

class SchedulingProblem(BaseModel):
    machines: List[Dict[str, int]]
    jobs: List[Dict[str, int]]
    rig_change_times: List[List[int]]
    solver_settings: Dict[str, Any]

# OWPy API Models based on OpenAPI spec
class SolverFunction(str, Enum):
    GLOBAL = "GLOBAL"
    TWO_STEP = "TWO_STEP"
    SA = "SA"
    BRANCH_AND_BOUND = "BranchAndBound"

class SolverSettings(BaseModel):
    MaxTime: int = Field(..., description="Maximum time in seconds allowed to run the solver")
    UseHeuristics: bool = Field(..., description="Whether to use heuristic to get faster results")

class IOScenario(BaseModel):
    Machines: List[Dict[str, int]] = Field(..., description="The list of machines available for the current shift")
    Jobs: List[Dict[str, int]] = Field(..., description="The list of jobs to be distributed and processed")
    Rigs: List[List[int]] = Field(..., description="The estimated time for changing a machine from one rig to another")

class ScenarioOptimizeRequest(BaseModel):
    scenarioInput: IOScenario
    solverFunc: SolverFunction
    solver_settings: SolverSettings

class SolutionAndTime(BaseModel):
    makespan: int = Field(..., description="Maximum time the machines have for job processing and rig changes")
    machines_distribution: Dict[str, List[int]] = Field(..., description="Processing sequence of jobs per machine")
    time_needed_in_s: int = Field(..., description="Time needed to solve the problem in seconds")

@app.get("/")
async def root():
    return {"message": "OWPy Chatbot API", "status": "running"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a chat message from the user and provide a response.

    The chatbot will:
    1. Analyze the user's message
    2. Attempt to formulate an optimization problem if applicable
    3. Make API calls to OWPy if a valid problem is formulated
    4. Refer to support team if the query is too complex
    """
    user_message = request.message
    context = request.context or {}

    # Process the user message to understand the intent
    intent, complexity = analyze_message(user_message)

    # If the query is too complex, refer to support
    if complexity > 7:  # On a scale of 1-10
        return ChatResponse(
            response="Your query appears to be complex. I recommend reaching out to our support team for assistance.",
            requires_support=True
        )

    # If the intent is to solve a scheduling problem
    if intent == "solve_optimization":
        try:
            # Formulate the scheduling problem
            problem = formulate_scheduling_problem(user_message, context)

            # Call OWPy API or generate random solution in test mode
            api_response = call_owpy_api(problem, request.test_mode)

            return ChatResponse(
                response=generate_response(user_message, api_response),
                scheduling_problem=problem.dict(),
                api_response=api_response
            )
        except ValueError as e:
            return ChatResponse(
                response=f"I couldn't formulate your scheduling problem: {str(e)}",
                requires_support=True
            )
        except Exception as e:
            return ChatResponse(
                response=f"An error occurred while processing your request: {str(e)}",
                requires_support=True
            )

    # For other intents, provide a general response
    return ChatResponse(
        response=generate_general_response(user_message, intent)
    )


def analyze_message(message: str) -> tuple:
    """
    Analyze the user message to determine intent and complexity.

    Returns:
        tuple: (intent, complexity)
    """
    # This is a simplified implementation
    # In a real-world scenario, you would use an NLP model or service

    # Check for scheduling-related keywords
    scheduling_keywords = ["schedule", "scheduling", "machine", "job", "rig", "makespan", "sequence", "assignment"]
    if any(keyword in message.lower() for keyword in scheduling_keywords):
        intent = "solve_optimization"  # Keep the same intent name for compatibility
    else:
        intent = "general_query"

    # Estimate complexity based on message length and presence of technical terms
    complexity = min(len(message) / 50, 10)  # Simple heuristic
    technical_terms = ["rig change times", "multiple machines", "precedence constraints", "release dates", "due dates"]
    for term in technical_terms:
        if term in message.lower():
            complexity += 1

    return intent, min(complexity, 10)


def call_gemini_api(prompt: str, user_message: str) -> Dict[str, Any]:
    """
    Call the Gemini API with the system prompt and user message.

    Args:
        prompt: The system prompt
        user_message: The user message

    Returns:
        Dict[str, Any]: The parsed JSON response from Gemini
    """
    try:
        # Load the Gemini 2.5 Pro Experimental model
        model = genai.GenerativeModel('gemini-2.5-pro-experimental')

        # Combine system prompt and user message
        full_prompt = f"{prompt}\n\nUser query: {user_message}"

        # Generate response from Gemini
        response = model.generate_content(full_prompt)

        # Extract the text from the response
        response_text = response.text

        # Parse the JSON response
        try:
            # Try to parse the response as JSON
            return json.loads(response_text)
        except json.JSONDecodeError:
            # If the response is not valid JSON, try to extract JSON from the text
            # This handles cases where the model might add explanatory text
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            else:
                raise ValueError("Could not parse JSON from Gemini response")

    except Exception as e:
        raise Exception(f"Error calling Gemini API: {str(e)}")


def formulate_scheduling_problem(message: str, context: Dict[str, Any]) -> SchedulingProblem:
    """
    Formulate a scheduling problem based on the user message using Gemini 2.5 Pro.

    Args:
        message: The user message describing the scheduling problem
        context: Additional context that may contain pre-formulated parts of the problem

    Returns:
        SchedulingProblem: The formulated scheduling problem
    """
    try:
        # If context already contains all the necessary components, use them directly
        if all(key in context for key in ["machines", "jobs", "rig_change_times", "solver_settings"]):
            return SchedulingProblem(
                machines=context["machines"],
                jobs=context["jobs"],
                rig_change_times=context["rig_change_times"],
                solver_settings=context["solver_settings"]
            )

        # Load the system prompt
        try:
            with open("system_prompt.txt", "r") as f:
                system_prompt = f.read()
        except FileNotFoundError:
            raise ValueError("System prompt file not found")

        # Call Gemini API to parse the message
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        # Call Gemini API
        problem_json = call_gemini_api(system_prompt, message)

        # Validate the response
        required_keys = ["machines", "jobs", "rig_change_times", "solver_settings"]
        if not all(key in problem_json for key in required_keys):
            missing_keys = [key for key in required_keys if key not in problem_json]
            raise ValueError(f"Missing required keys in Gemini response: {missing_keys}")

        # Create and return the SchedulingProblem
        return SchedulingProblem(
            machines=problem_json["machines"],
            jobs=problem_json["jobs"],
            rig_change_times=problem_json["rig_change_times"],
            solver_settings=problem_json["solver_settings"]
        )

    except Exception as e:
        # If there's an error with Gemini, fall back to the context or default values
        print(f"Error formulating problem with Gemini: {str(e)}")

        # Use default values for scheduling problem
        machines = context.get("machines", [
            {"machine_id": 1, "processing_time": 1}
        ])

        jobs = context.get("jobs", [
            {"job_id": 1, "rig_id": 1},
            {"job_id": 2, "rig_id": 1}
        ])

        rig_change_times = context.get("rig_change_times", [
            [0, 1],
            [1, 0]
        ])

        solver_settings = context.get("solver_settings", {
            "max_time": 60,
            "use_heuristics": True,
            "solver_function": "GLOBAL"
        })

        return SchedulingProblem(
            machines=machines,
            jobs=jobs,
            rig_change_times=rig_change_times,
            solver_settings=solver_settings
        )


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
    import random
    from datetime import datetime

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


def generate_response(message: str, api_response: Dict[str, Any]) -> str:
    """
    Generate a human-friendly response based on the API response.

    Args:
        message: The original user message
        api_response: The response from the OWPy API

    Returns:
        str: A human-friendly response
    """
    if api_response["status"] == "success":
        solution = api_response["solution"]
        makespan = solution["objective_value"]  # This is the makespan in scheduling problems
        job_assignments = solution["variables"]
        solve_time = solution.get("solve_time", "unknown")

        response = f"I've solved your scheduling problem! The optimal makespan is {makespan} time units.\n\n"

        response += "The job assignments to machines are:\n"

        # Group jobs by machine for clearer presentation
        jobs_by_machine = {}
        for job_name, machine_id in job_assignments.items():
            if machine_id not in jobs_by_machine:
                jobs_by_machine[machine_id] = []
            jobs_by_machine[machine_id].append(job_name)

        for machine_id, jobs in jobs_by_machine.items():
            response += f"- Machine {machine_id}: {', '.join(jobs)}\n"

        response += f"\nThe solution was found in {solve_time} seconds."

        return response
    elif api_response["status"] == "error":
        error_message = api_response.get("error", "Unknown error")
        return f"I couldn't solve your scheduling problem. The API returned the following error: {error_message}\n\nPlease check your formulation or contact support."
    else:
        return "I couldn't solve your scheduling problem. Please check your formulation or contact support."


def generate_general_response(message: str, intent: str) -> str:
    """
    Generate a general response for non-scheduling queries.

    Args:
        message: The user message
        intent: The detected intent

    Returns:
        str: A response message
    """
    if intent == "general_query":
        return (
            "I'm an OWPy assistant designed to help with scheduling problems. "
            "You can ask me to formulate and solve scheduling problems for you. "
            "For example, you could describe a problem with 'machines, jobs, and rig requirements' "
            "and I'll help you find the optimal schedule to minimize the makespan."
        )

    return "I'm not sure how to help with that. Could you provide more details about your scheduling problem?"
