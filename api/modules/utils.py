from typing import Dict, Any, Tuple

def analyze_message(message: str) -> Tuple[str, float]:
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