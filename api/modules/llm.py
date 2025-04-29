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

# Configure logging
logger = logging.getLogger("api.llm")

# Configure Google Generative AI with API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    logger.info("Google Generative AI configured with API key")
else:
    logger.warning("GOOGLE_API_KEY not found in environment variables")

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
                "processing_time": random.randint(1, 5)
            })

        # Generate jobs
        jobs = []
        for i in range(1, num_jobs + 1):
            jobs.append({
                "job_id": i,
                "rig_id": random.randint(1, num_rigs)
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
            "machines": [{"machine_id": 1, "processing_time": 1}],
            "jobs": [{"job_id": 1, "rig_id": 1}],
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
