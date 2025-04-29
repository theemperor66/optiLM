from typing import Dict, Any, Tuple
import logging
import time
import traceback

# Configure logging
logger = logging.getLogger("api.utils")

def analyze_message(message: str) -> Tuple[str, float]:
    """
    Analyze the user message to determine intent and complexity.

    Args:
        message: The user message to analyze

    Returns:
        tuple: (intent, complexity)

    Raises:
        ValueError: If the message is empty or invalid
    """
    request_id = f"analyze-{int(time.time())}"
    logger.info(f"Analyzing message {request_id} (length: {len(message)})")
    start_time = time.time()

    try:
        # Validate input
        if not message or not isinstance(message, str):
            logger.error(f"Invalid message for request {request_id}: empty or not a string")
            raise ValueError("Message must be a non-empty string")

        # This is a simplified implementation
        # In a real-world scenario, you would use an NLP model or service

        # Check for scheduling-related keywords
        scheduling_keywords = ["schedule", "scheduling", "machine", "job", "rig", "makespan", "sequence", "assignment"]
        keyword_matches = [keyword for keyword in scheduling_keywords if keyword in message.lower()]

        if keyword_matches:
            intent = "solve_optimization"  # Keep the same intent name for compatibility
            logger.debug(f"Detected scheduling intent for request {request_id} based on keywords: {', '.join(keyword_matches)}")
        else:
            intent = "general_query"
            logger.debug(f"Detected general query intent for request {request_id}")

        # Estimate complexity based on message length and presence of technical terms
        length_complexity = min(len(message) / 200, 10)  # Simple heuristic - 1 point per 200 characters
        logger.debug(f"Base complexity for request {request_id} from message length ({len(message)} chars): {length_complexity:.2f}")

        complexity = length_complexity
        technical_terms = ["rig change times", "multiple machines", "precedence constraints", "release dates", "due dates"]
        term_matches = [term for term in technical_terms if term in message.lower()]

        for term in term_matches:
            complexity += 1
            logger.debug(f"Increased complexity for request {request_id} due to term: {term}")

        final_complexity = min(complexity, 10)

        if term_matches:
            logger.debug(f"Added {len(term_matches)} points to complexity for request {request_id} due to technical terms: {', '.join(term_matches)}")

        process_time = time.time() - start_time
        logger.info(f"Message analysis for request {request_id} completed in {process_time:.3f}s: intent={intent}, complexity={final_complexity:.1f}")

        return intent, final_complexity

    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Error analyzing message for request {request_id} after {process_time:.3f}s: {str(e)}")
        logger.error(traceback.format_exc())
        raise ValueError(f"Failed to analyze message: {str(e)}")

def generate_response(message: str, api_response: Dict[str, Any]) -> str:
    """
    Generate a human-friendly response based on the API response.

    Args:
        message: The original user message
        api_response: The response from the OWPy API

    Returns:
        str: A human-friendly response
    """
    request_id = f"response-{int(time.time())}"
    logger.info(f"Generating response for request {request_id}")
    start_time = time.time()

    try:
        # Validate input
        if not isinstance(api_response, dict):
            logger.error(f"Invalid API response for request {request_id}: not a dictionary")
            return "I received an invalid response format. Please try again or contact support."

        # Check response status
        status = api_response.get("status")
        logger.debug(f"API response status for request {request_id}: {status}")

        if status == "success":
            # Extract solution details
            if "solution" not in api_response:
                logger.error(f"Missing solution in API response for request {request_id}")
                return "I received an incomplete response. Please try again or contact support."

            solution = api_response["solution"]

            # Extract and validate solution components
            if "objective_value" not in solution:
                logger.error(f"Missing objective_value in solution for request {request_id}")
                return "The solution is missing the objective value. Please try again or contact support."

            makespan = solution["objective_value"]  # This is the makespan in scheduling problems

            if "variables" not in solution:
                logger.error(f"Missing variables in solution for request {request_id}")
                return "The solution is missing the job assignments. Please try again or contact support."

            job_assignments = solution["variables"]
            solve_time = solution.get("solve_time", "unknown")

            logger.debug(f"Solution details for request {request_id}: makespan={makespan}, jobs={len(job_assignments)}, solve_time={solve_time}")

            # Build the response
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

            process_time = time.time() - start_time
            logger.info(f"Generated success response for request {request_id} in {process_time:.3f}s")

            return response

        elif status == "error":
            error_message = api_response.get("error", "Unknown error")
            status_code = api_response.get("status_code", "")

            logger.warning(f"Error in API response for request {request_id}: {error_message} (status_code: {status_code})")

            error_response = f"I couldn't solve your scheduling problem. "
            error_response += f"Error: {error_message}\n\n"

            # Add more helpful guidance based on error type
            if "authentication" in error_message.lower() or "credentials" in error_message.lower():
                error_response += "This appears to be an authentication issue. Please check your API credentials."
            elif "timeout" in error_message.lower():
                error_response += "The solver timed out. Try simplifying your problem or increasing the solver time limit."
            elif "format" in error_message.lower() or "invalid" in error_message.lower():
                error_response += "There might be an issue with how the problem was formulated. Please check your input data."
            else:
                error_response += "Please check your formulation or contact support for assistance."

            process_time = time.time() - start_time
            logger.info(f"Generated error response for request {request_id} in {process_time:.3f}s")

            return error_response

        else:
            logger.warning(f"Unknown status in API response for request {request_id}: {status}")
            return "I couldn't solve your scheduling problem due to an unknown response status. Please try again or contact support."

    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Error generating response for request {request_id} after {process_time:.3f}s: {str(e)}")
        logger.error(traceback.format_exc())
        return f"I encountered an error while processing the solution: {str(e)}. Please try again or contact support."

def generate_general_response(message: str, intent: str) -> str:
    """
    Generate a general response for non-scheduling queries.

    Args:
        message: The user message
        intent: The detected intent

    Returns:
        str: A response message
    """
    request_id = f"general-{int(time.time())}"
    logger.info(f"Generating general response for request {request_id} with intent: {intent}")
    start_time = time.time()

    try:
        # Validate inputs
        if not message:
            logger.warning(f"Empty message for request {request_id}")

        if not intent:
            logger.warning(f"Empty intent for request {request_id}")
            intent = "unknown"

        # Generate appropriate response based on intent
        if intent == "general_query":
            response = (
                "I'm an OWPy assistant designed to help with scheduling problems. "
                "You can ask me to formulate and solve scheduling problems for you. "
                "For example, you could describe a problem with 'machines, jobs, and rig requirements' "
                "and I'll help you find the optimal schedule to minimize the makespan."
            )
            logger.debug(f"Generated general information response for request {request_id}")
        elif intent == "solve_optimization" and len(message) < 20:
            # If the intent is to solve a problem but the message is too short
            response = (
                "I'd be happy to help solve your scheduling problem, but I need more details. "
                "Please describe your problem with information about machines, jobs, and any constraints. "
                "For example: 'I have 3 machines and 10 jobs with different rig requirements. How should I schedule them?'"
            )
            logger.debug(f"Generated request for more details response for request {request_id}")
        else:
            response = "I'm not sure how to help with that. Could you provide more details about your scheduling problem?"
            logger.debug(f"Generated fallback response for request {request_id} with intent: {intent}")

        process_time = time.time() - start_time
        logger.info(f"General response generated for request {request_id} in {process_time:.3f}s")

        return response

    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Error generating general response for request {request_id} after {process_time:.3f}s: {str(e)}")
        logger.error(traceback.format_exc())
        return "I'm having trouble generating a response right now. Please try again with a more specific question about scheduling problems."
