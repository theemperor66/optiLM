import google.generativeai as genai
import json
import os
import re
import random
import logging
import time
import traceback
from typing import Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .models import LLMReply

# Configure logging
logger = logging.getLogger("api.llm")

# Configure Google Generative AI with API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    logger.info("Google Generative AI configured with API key")
else:
    logger.warning("GOOGLE_API_KEY not found in environment variables")

def load_prompt(filename: str) -> str:
    """
    Load a prompt template from the prompts directory.

    Args:
        filename: The name of the prompt file to load

    Returns:
        str: The content of the prompt file
    """
    try:
        prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", filename)
        with open(prompt_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Prompt file {filename} not found")
        raise ValueError(f"Prompt file {filename} not found")

def call_builder(user_msg: str, state: Dict, confirm=False) -> LLMReply:
    """
    Call the interactive builder to process a user message and update the problem state.

    Args:
        user_msg: The user message
        state: The current problem state
        confirm: Whether to use the confirm and solve prompt

    Returns:
        LLMReply: The response from the LLM
    """
    tmpl = "02_confirm_and_solve.txt" if confirm else "01_interactive_builder.txt"

    # Load prompt templates
    header_prompt = load_prompt("00_context_header.txt")
    template_prompt = load_prompt(tmpl)

    # Replace placeholders manually to avoid issues with unescaped curly braces
    state_json_str = json.dumps(state or {}, separators=(',',':'))
    template_prompt = template_prompt.replace("{state_json}", state_json_str)
    template_prompt = template_prompt.replace("{user_msg}", user_msg)

    prompt = header_prompt + template_prompt

    try:
        # Call Gemini API
        raw = _call_gemini(prompt)

        # Strip markdown fences if present
        clean = re.sub(r"^\s*```json\s*|\s*```\s*$", "", raw.strip(), flags=re.I|re.S)

        # Parse the response as JSON
        return LLMReply.model_validate_json(clean)
    except Exception as e:
        logger.error(f"LLM error: {e}\nRaw output: {raw[:120] if 'raw' in locals() else 'No output'}")
        return LLMReply(
            scheduling_problem=state or {},
            clarification_question="Sorry, I couldn't parse that. Could you rephrase?",
            is_complete=False,
        )

def _call_gemini(prompt: str) -> str:
    """
    Call the Gemini API with a prompt and return the raw text response.

    Args:
        prompt: The prompt to send to the Gemini API

    Returns:
        str: The raw text response from the Gemini API
    """
    if not GOOGLE_API_KEY:
        logger.error("Cannot call Gemini API: API key not configured")
        raise ValueError("Google API key not configured. Please set the GOOGLE_API_KEY environment variable.")

    request_id = f"gemini-{int(time.time())}"
    logger.info(f"Calling Gemini API (request {request_id})")

    try:
        # Load the Gemini 2.5 Pro Experimental model
        model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')

        # Generate response from Gemini
        response = model.generate_content(prompt)

        # Extract the text from the response
        if not hasattr(response, 'text') or not response.text:
            logger.error(f"Invalid response format from Gemini API for request {request_id}")
            raise ValueError("Invalid response format from Gemini API: missing text field")

        return response.text
    except Exception as e:
        logger.error(f"Error calling Gemini API: {str(e)}")
        raise

def _mock_llm_reply(msg: str, current_state: Dict, confirm: bool) -> LLMReply:
    """
    Generate a mock LLM reply for testing without calling the Gemini API.

    Args:
        msg: The user message
        current_state: The current problem state
        confirm: Whether this is a confirmation step

    Returns:
        LLMReply: A mock LLM reply
    """
    # Initialize with current state or empty dict
    state = current_state.copy() if current_state else {}

    # For testing purposes, let's create a deterministic sequence of responses
    # based on the number of messages exchanged

    # Step 1: First message about machines
    if "machine" in msg.lower() and "machines" not in state:
        state["machines"] = [{"machine_id": 1, "start_rig_id": 1}, {"machine_id": 2, "start_rig_id": 1}]
        return LLMReply(
            scheduling_problem=state,
            clarification_question="I've added 2 machines. Can you tell me about the jobs?",
            is_complete=False,
            ready_to_solve=False
        )

    # Step 2: Message about jobs
    if "job" in msg.lower() and "jobs" not in state and "machines" in state:
        state["jobs"] = [
            {"job_id": 1, "rig_id": 1, "processing_time": 3},
            {"job_id": 2, "rig_id": 2, "processing_time": 4}
        ]
        return LLMReply(
            scheduling_problem=state,
            clarification_question="I've added the jobs. Can you provide the rig change times matrix?",
            is_complete=False,
            ready_to_solve=False
        )

    # Step 3: Message about rig matrix
    if ("rig matrix" in msg.lower() or "change time" in msg.lower()) and "rig_change_times" not in state and "jobs" in state:
        state["rig_change_times"] = [[0, 1], [1, 0]]

        # Add solver settings
        state["solver_settings"] = {
            "max_time": 30,
            "use_heuristics": True,
            "solver_function": "GLOBAL"
        }

        return LLMReply(
            scheduling_problem=state,
            clarification_question="I've added the rig change times and solver settings. The problem is complete. Would you like me to solve it now?",
            is_complete=True,
            ready_to_solve=False
        )

    # Step 4: Solve the problem
    if confirm and "solve" in msg.lower() and all(k in state for k in ["machines", "jobs", "rig_change_times", "solver_settings"]):
        return LLMReply(
            scheduling_problem=state,
            clarification_question=None,
            is_complete=True,
            ready_to_solve=True
        )

    # Default response for other cases or if the sequence is broken
    return LLMReply(
        scheduling_problem=state,
        clarification_question="I'm not sure what you're asking. Can you provide information about machines, jobs, or rig change times?",
        is_complete=False,
        ready_to_solve=False
    )

def generate_mock_response(user_message: str) -> Dict[str, Any]:
    """
    Generate a mock response for testing without calling the Gemini API.

    Args:
        user_message: The user message

    Returns:
        Dict[str, Any]: A mock response that mimics the Gemini API response
    """
    logger.info("Generating mock response for testing")
    start_time = time.time()

    try:
        # Default values
        num_machines = 2
        num_jobs = 3
        num_rigs = 2

        # Try to extract numbers from the message
        machines_match = re.search(r'(\d+)\s*machines?', user_message.lower())
        if machines_match:
            num_machines = int(machines_match.group(1))
            logger.debug(f"Extracted {num_machines} machines from message")

        jobs_match = re.search(r'(\d+)\s*jobs?', user_message.lower())
        if jobs_match:
            num_jobs = int(jobs_match.group(1))
            logger.debug(f"Extracted {num_jobs} jobs from message")

        rigs_match = re.search(r'(\d+)\s*rigs?', user_message.lower())
        if rigs_match:
            num_rigs = int(rigs_match.group(1))
            logger.debug(f"Extracted {num_rigs} rigs from message")

        # Generate machines
        machines = []
        for i in range(1, num_machines + 1):
            machines.append({
                "machine_id": i,
                "start_rig_id": random.randint(1, num_rigs)
            })

        # Generate jobs
        jobs = []
        for i in range(1, num_jobs + 1):
            jobs.append({
                "job_id": i,
                "rig_id": random.randint(1, num_rigs),
                "processing_time": random.randint(1, 5)
            })

        # Generate rig change times
        rig_change_times = []
        for i in range(num_rigs):
            row = []
            for j in range(num_rigs):
                if i == j:
                    row.append(0)  # No change time for same rig
                else:
                    row.append(random.randint(1, 3))  # Random change time
            rig_change_times.append(row)

        # Generate solver settings
        solver_functions = ["GLOBAL", "TWO_STEP", "SA", "BranchAndBound"]
        solver_settings = {
            "max_time": 60,
            "use_heuristics": True,
            "solver_function": random.choice(solver_functions)
        }

        # Create the mock response
        mock_response = {
            "machines": machines,
            "jobs": jobs,
            "rig_change_times": rig_change_times,
            "solver_settings": solver_settings
        }

        process_time = time.time() - start_time
        logger.info(f"Mock response generated in {process_time:.3f}s")
        logger.debug(f"Mock response: {mock_response}")

        return mock_response

    except Exception as e:
        logger.error(f"Error generating mock response: {str(e)}")
        logger.error(traceback.format_exc())
        # Return a minimal valid response in case of error
        return {
            "machines": [{"machine_id": 1}],
            "jobs": [{"job_id": 1, "rig_id": 1, "processing_time": 1}],
            "rig_change_times": [[0]],
            "solver_settings": {"max_time": 60, "use_heuristics": True, "solver_function": "GLOBAL"}
        }

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((json.JSONDecodeError, ValueError))
)
def call_gemini_api(prompt: str, user_message: str, test_mode: bool = False) -> Dict[str, Any]:
    """
    Call the Gemini API with the system prompt and user message.

    This function includes retry logic for handling transient errors.
    Note: test_mode parameter is kept for compatibility but no longer affects this function.

    Args:
        prompt: The system prompt
        user_message: The user message
        test_mode: Deprecated parameter, kept for compatibility

    Returns:
        Dict[str, Any]: The parsed JSON response from Gemini

    Raises:
        Exception: If the API call fails after retries or if the response cannot be parsed
    """
    # test_mode is ignored as we always want to call the real Gemini API
    if test_mode:
        logger.info("Test mode enabled, but still using real Gemini API for parsing")

    # Check if API key is configured
    if not GOOGLE_API_KEY:
        logger.error("Cannot call Gemini API: API key not configured")
        raise ValueError("Google API key not configured. Please set the GOOGLE_API_KEY environment variable.")

    request_id = f"gemini-{int(time.time())}"
    logger.info(f"Calling Gemini API (request {request_id})")
    start_time = time.time()

    try:
        # Load the Gemini 2.5 Pro Experimental model
        logger.debug(f"Loading Gemini model for request {request_id}")
        model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')

        # Combine system prompt and user message
        full_prompt = f"{prompt}\n\nUser query: {user_message}"
        logger.debug(f"Prompt length for request {request_id}: {len(full_prompt)} characters")

        # Generate response from Gemini
        logger.debug(f"Sending request {request_id} to Gemini API")
        response = model.generate_content(full_prompt)

        # Extract the text from the response
        if not hasattr(response, 'text') or not response.text:
            logger.error(f"Invalid response format from Gemini API for request {request_id}")
            raise ValueError("Invalid response format from Gemini API: missing text field")

        response_text = response.text
        logger.debug(f"Received response for request {request_id}, length: {len(response_text)} characters")

        # Parse the JSON response
        try:
            # Try to parse the response as JSON
            logger.debug(f"Attempting to parse response as JSON for request {request_id}")
            parsed_response = json.loads(response_text)

            # Validate the response structure
            validate_response_structure(parsed_response)

            process_time = time.time() - start_time
            logger.info(f"Successfully processed Gemini API response for request {request_id} in {process_time:.3f}s")
            return parsed_response

        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error for request {request_id}: {str(e)}")
            logger.debug(f"Raw response: {response_text[:500]}...")

            # If the response is not valid JSON, try to extract JSON from the text
            # This handles cases where the model might add explanatory text
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                logger.info(f"Found JSON in code block for request {request_id}")
                extracted_json = json_match.group(1)
                parsed_response = json.loads(extracted_json)

                # Validate the response structure
                validate_response_structure(parsed_response)

                process_time = time.time() - start_time
                logger.info(f"Successfully extracted and parsed JSON from Gemini API response for request {request_id} in {process_time:.3f}s")
                return parsed_response
            else:
                logger.error(f"Could not find JSON in response for request {request_id}")
                logger.debug(f"Response text: {response_text[:500]}...")
                raise ValueError(f"Could not parse JSON from Gemini response. The model returned: {response_text[:100]}...")

    except Exception as e:
        process_time = time.time() - start_time
        # Check if this is a Gemini API error by examining the exception message or class
        if "genai" in str(e.__class__).lower() or "google" in str(e.__class__).lower():
            logger.error(f"Gemini API generation error for request {request_id} after {process_time:.3f}s: {str(e)}")
            raise ValueError(f"Gemini API generation error: {str(e)}. Please try again with a different query.")
        else:
            logger.error(f"Unexpected error calling Gemini API for request {request_id} after {process_time:.3f}s: {str(e)}")
            logger.error(traceback.format_exc())
            raise Exception(f"Error calling Gemini API: {str(e)}. Please check your query and try again.")

def validate_response_structure(response: Dict[str, Any]) -> None:
    """
    Validate that the response has the expected structure.

    Args:
        response: The parsed JSON response

    Raises:
        ValueError: If the response is missing required fields
    """
    required_fields = ["machines", "jobs", "rig_change_times", "solver_settings"]
    missing_fields = [field for field in required_fields if field not in response]

    if missing_fields:
        missing_fields_str = ", ".join(missing_fields)
        logger.error(f"Response missing required fields: {missing_fields_str}")
        raise ValueError(f"Invalid response format: missing required fields ({missing_fields_str})")

    # Validate machines
    if not isinstance(response["machines"], list):
        logger.error("Invalid machines format: not a list")
        raise ValueError("Invalid response format: machines must be a list")

    # Validate jobs
    if not isinstance(response["jobs"], list):
        logger.error("Invalid jobs format: not a list")
        raise ValueError("Invalid response format: jobs must be a list")

    # Validate rig_change_times
    if not isinstance(response["rig_change_times"], list):
        logger.error("Invalid rig_change_times format: not a list")
        raise ValueError("Invalid response format: rig_change_times must be a list")

    # Validate solver_settings
    if not isinstance(response["solver_settings"], dict):
        logger.error("Invalid solver_settings format: not a dictionary")
        raise ValueError("Invalid response format: solver_settings must be a dictionary")
