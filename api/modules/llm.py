import json
import os
import re
import random
import logging
import time
import traceback
from typing import Dict, Any, Optional, List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .models import LLMReply
from .llm_providers import get_provider, get_available_providers

# Configure logging
logger = logging.getLogger("api.llm")

# Get the selected LLM provider
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
logger.info(f"Using LLM provider: {LLM_PROVIDER}")

# Get the available providers
available_providers = get_available_providers()
logger.info(f"Available LLM providers: {', '.join(available_providers)}")

# Initialize the provider
try:
    provider = get_provider(LLM_PROVIDER)
    logger.info(f"LLM provider '{LLM_PROVIDER}' initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize LLM provider '{LLM_PROVIDER}': {str(e)}")
    logger.warning(f"Falling back to default provider 'gemini'")
    provider = get_provider("gemini")


class MessageHistory:
    """
    Container for message history in conversations with the LLM.
    """
    
    def __init__(self, max_history: int = 10):
        """
        Initialize a message history container.
        
        Args:
            max_history: Maximum number of messages to keep in history
        """
        self.messages: List[Dict[str, str]] = []
        self.max_history = max_history
    
    def add_user_message(self, content: str) -> None:
        """
        Add a user message to the history.
        
        Args:
            content: The content of the user message
        """
        self.messages.append({"role": "user", "content": content})
        self._truncate_if_needed()
    
    def add_assistant_message(self, content: str) -> None:
        """
        Add an assistant message to the history.
        
        Args:
            content: The content of the assistant message
        """
        self.messages.append({"role": "assistant", "content": content})
        self._truncate_if_needed()
    
    def clear(self) -> None:
        """
        Clear the message history.
        """
        self.messages = []
    
    def _truncate_if_needed(self) -> None:
        """
        Truncate the history if it exceeds max_history.
        """
        if len(self.messages) > self.max_history:
            # Remove oldest messages, but try to keep user/assistant pairs
            # This helps maintain context even when truncating
            truncate_count = len(self.messages) - self.max_history
            self.messages = self.messages[truncate_count:]

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

def call_builder(user_msg: str, state: Dict, message_history: Optional[MessageHistory] = None, confirm=False) -> LLMReply:
    """
    Call the interactive builder to process a user message and update the problem state.

    Args:
        user_msg: The user message
        state: The current problem state
        message_history: Optional message history for conversation context
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
    
    # Initialize message history if not provided
    if message_history is None:
        message_history = MessageHistory()
    
    # Add the current user message to history
    message_history.add_user_message(user_msg)

    # Add an explicit instruction for JSON output
    if LLM_PROVIDER == "groq":
        json_instruction = "\n\nIMPORTANT: Your response MUST be valid JSON format ONLY, without any explanations or text outside the JSON structure."
        prompt = header_prompt + template_prompt + json_instruction
    else:
        prompt = header_prompt + template_prompt

    try:
        # Use the message history for context
        if LLM_PROVIDER == "groq":
            # Add JSON instruction for Groq
            json_instruction = "\n\nIMPORTANT: Your response MUST be valid JSON format ONLY, without any explanations or text outside the JSON structure."
            system_prompt = header_prompt + template_prompt + json_instruction
        else:
            system_prompt = header_prompt + template_prompt
            
        # Call API with message history
        raw = provider.generate_json_with_history(system_prompt, message_history.messages)
        
        # Convert the raw JSON to LLMReply
        response = LLMReply.model_validate(raw)
        
        # Add the assistant's response to history (as JSON string for consistency)
        message_history.add_assistant_message(json.dumps(raw))
        
        return response
    except Exception as e:
        logger.error(f"LLM error: {e}\nRaw output: {str(raw)[:120] if 'raw' in locals() else 'No output'}")
        return LLMReply(
            scheduling_problem=state or {},
            clarification_question="Sorry, I couldn't parse that. Could you rephrase?",
            is_complete=False,
        )

def extract_json_from_response(text: str) -> str:
    """
    Extract JSON from the model's response, handling various ways the model might format it.
    """
    # First try: Standard markdown code block extraction
    json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()
    
    # Second try: Find the first opening curly brace and its matching closing brace
    start_idx = text.find('{')
    if start_idx >= 0:
        # Find matching closing brace by counting balanced braces
        brace_count = 0
        in_string = False
        escape_next = False
        
        for i in range(start_idx, len(text)):
            char = text[i]
            if escape_next:
                escape_next = False
                continue
            if char == '\\' and not escape_next:
                escape_next = True
            elif char == '"' and not escape_next:
                in_string = not in_string
            elif not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return text[start_idx:i+1].strip()
    
    # If all extraction attempts fail, return the original stripped text
    # This may still fail JSON parsing, but our try/except will handle it
    return text.strip()

def _call_gemini(prompt: str) -> str:
    """
    Call the selected LLM provider with a prompt and return the raw text response.

    Args:
        prompt: The prompt to send to the LLM

    Returns:
        str: The raw text response from the LLM
    """
    request_id = f"llm-{int(time.time())}"
    logger.info(f"Calling {LLM_PROVIDER} provider (request {request_id})")

    try:
        # Generate text using the selected provider
        return provider.generate_text(prompt)
    except Exception as e:
        logger.error(f"Error calling {LLM_PROVIDER} provider: {str(e)}")
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
    Call the selected LLM provider with the system prompt and user message.

    This function includes retry logic for handling transient errors.
    Note: test_mode parameter is kept for compatibility but no longer affects this function.

    Args:
        prompt: The system prompt
        user_message: The user message
        test_mode: Deprecated parameter, kept for compatibility

    Returns:
        Dict[str, Any]: The parsed JSON response from the LLM

    Raises:
        Exception: If the API call fails after retries or if the response cannot be parsed
    """
    # test_mode is ignored as we always want to call the real LLM API for parsing
    if test_mode:
        logger.info(f"Test mode enabled, but still using real {LLM_PROVIDER} API for parsing")

    request_id = f"llm-{int(time.time())}"
    logger.info(f"Calling {LLM_PROVIDER} API (request {request_id})")
    start_time = time.time()

    try:
        # Generate and parse JSON using the selected provider
        parsed_response = provider.generate_json(prompt, user_message)

        # Validate the response structure
        validate_response_structure(parsed_response)

        process_time = time.time() - start_time
        logger.info(f"Successfully processed {LLM_PROVIDER} API response for request {request_id} in {process_time:.3f}s")
        return parsed_response

    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Error calling {LLM_PROVIDER} API for request {request_id} after {process_time:.3f}s: {str(e)}")
        logger.error(traceback.format_exc())
        raise Exception(f"Error calling {LLM_PROVIDER} API: {str(e)}. Please check your query and try again.")

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
