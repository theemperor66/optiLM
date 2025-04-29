import os
from typing import Dict, Any
from .models import SchedulingProblem
from .llm import call_gemini_api

def formulate_scheduling_problem(message: str, context: Dict[str, Any], test_mode: bool = False) -> SchedulingProblem:
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
            import os.path
            prompt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "system_prompt.txt")
            with open(prompt_path, "r") as f:
                system_prompt = f.read()
        except FileNotFoundError:
            raise ValueError("System prompt file not found")

        # Call Gemini API to parse the message
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        # Call Gemini API or generate mock response in test mode
        problem_json = call_gemini_api(system_prompt, message, test_mode)

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
