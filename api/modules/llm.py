import google.generativeai as genai
import json
import os
import re
from typing import Dict, Any

# Configure Google Generative AI with API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    print("Warning: GOOGLE_API_KEY not found in environment variables")

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
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            else:
                raise ValueError("Could not parse JSON from Gemini response")

    except Exception as e:
        raise Exception(f"Error calling Gemini API: {str(e)}")