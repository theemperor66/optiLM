"""
This module provides an abstraction layer for different LLM providers.
It defines a common interface that all providers must implement and
a factory function to create the appropriate provider based on configuration.
"""

import os
import logging
import abc
from typing import Dict, Any, Optional, Type, List

# Configure logging
logger = logging.getLogger("api.llm_providers")

class LLMProvider(abc.ABC):
    """
    Abstract base class for LLM providers.
    All provider implementations must inherit from this class and implement its methods.
    """
    
    @abc.abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the provider with the necessary configuration.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def generate_text(self, prompt: str) -> str:
        """
        Generate text based on the given prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            str: The generated text
        """
        pass
    
    @abc.abstractmethod
    def generate_json(self, prompt: str, user_message: str) -> Dict[str, Any]:
        """
        Generate and parse JSON based on the given prompt and user message.
        
        Args:
            prompt: The system prompt
            user_message: The user message
            
        Returns:
            Dict[str, Any]: The parsed JSON response
        """
        pass

class GeminiProvider(LLMProvider):
    """
    Implementation of LLMProvider for Google's Gemini API.
    """
    
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.model = os.getenv("LLM_MODEL", "gemini-2.5-pro-exp-03-25")
        self.client = None
    
    def initialize(self) -> bool:
        """
        Initialize the Gemini provider with the API key.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            import google.generativeai as genai
            
            if not self.api_key:
                logger.error("Cannot initialize Gemini: API key not configured")
                return False
            
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model)
            
            logger.info("Gemini provider initialized successfully")
            logger.info(f"Using Gemini model: {self.model}")
            return True
        except ImportError:
            logger.error("Failed to import google.generativeai. Make sure it's installed.")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Gemini provider: {str(e)}")
            return False
    
    def generate_text(self, prompt: str) -> str:
        """
        Generate text using the Gemini API.
        
        Args:
            prompt: The prompt to send to Gemini
            
        Returns:
            str: The generated text
        """
        if not self.client:
            if not self.initialize():
                raise ValueError("Gemini provider not initialized")
        
        try:
            response = self.client.generate_content(prompt)
            
            if not hasattr(response, 'text') or not response.text:
                raise ValueError("Invalid response format from Gemini API: missing text field")
            
            return response.text
        except Exception as e:
            logger.error(f"Error generating text with Gemini: {str(e)}")
            raise
    
    def generate_json(self, prompt: str, user_message: str) -> Dict[str, Any]:
        """
        Generate and parse JSON using the Gemini API.
        
        Args:
            prompt: The system prompt
            user_message: The user message
            
        Returns:
            Dict[str, Any]: The parsed JSON response
        """
        import json
        import re
        
        full_prompt = f"{prompt}\n\nUser query: {user_message}"
        
        try:
            response_text = self.generate_text(full_prompt)
            
            # Try to parse the response as JSON
            try:
                parsed_response = json.loads(response_text)
                return parsed_response
            except json.JSONDecodeError:
                # If the response is not valid JSON, try to extract JSON from the text
                json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
                if json_match:
                    extracted_json = json_match.group(1)
                    parsed_response = json.loads(extracted_json)
                    return parsed_response
                else:
                    raise ValueError(f"Could not parse JSON from response: {response_text[:100]}...")
        except Exception as e:
            logger.error(f"Error generating JSON with Gemini: {str(e)}")
            raise

class OpenAIProvider(LLMProvider):
    """
    Implementation of LLMProvider for OpenAI's API.
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.client = None
    
    def initialize(self) -> bool:
        """
        Initialize the OpenAI provider with the API key.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            from openai import OpenAI
            
            if not self.api_key:
                logger.error("Cannot initialize OpenAI: API key not configured")
                return False
            
            self.client = OpenAI(api_key=self.api_key)
            
            logger.info("OpenAI provider initialized successfully")
            logger.info(f"Using OpenAI model: {self.model}")
            return True
        except ImportError:
            logger.error("Failed to import openai. Make sure it's installed.")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI provider: {str(e)}")
            return False
    
    def generate_text(self, prompt: str) -> str:
        """
        Generate text using the OpenAI API.
        
        Args:
            prompt: The prompt to send to OpenAI
            
        Returns:
            str: The generated text
        """
        if not self.client:
            if not self.initialize():
                raise ValueError("OpenAI provider not initialized")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            
            if not response.choices or not response.choices[0].message.content:
                raise ValueError("Invalid response format from OpenAI API: missing content")
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {str(e)}")
            raise
    
    def generate_json(self, prompt: str, user_message: str) -> Dict[str, Any]:
        """
        Generate and parse JSON using the OpenAI API.
        
        Args:
            prompt: The system prompt
            user_message: The user message
            
        Returns:
            Dict[str, Any]: The parsed JSON response
        """
        import json
        
        if not self.client:
            if not self.initialize():
                raise ValueError("OpenAI provider not initialized")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            if not response.choices or not response.choices[0].message.content:
                raise ValueError("Invalid response format from OpenAI API: missing content")
            
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            logger.error(f"Error generating JSON with OpenAI: {str(e)}")
            raise

class AnthropicProvider(LLMProvider):
    """
    Implementation of LLMProvider for Anthropic's Claude API.
    """
    
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229")
        self.client = None
    
    def initialize(self) -> bool:
        """
        Initialize the Anthropic provider with the API key.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            import anthropic
            
            if not self.api_key:
                logger.error("Cannot initialize Anthropic: API key not configured")
                return False
            
            self.client = anthropic.Anthropic(api_key=self.api_key)
            
            logger.info("Anthropic provider initialized successfully")
            logger.info(f"Using Anthropic model: {self.model}")
            return True
        except ImportError:
            logger.error("Failed to import anthropic. Make sure it's installed.")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic provider: {str(e)}")
            return False
    
    def generate_text(self, prompt: str) -> str:
        """
        Generate text using the Anthropic API.
        
        Args:
            prompt: The prompt to send to Anthropic
            
        Returns:
            str: The generated text
        """
        if not self.client:
            if not self.initialize():
                raise ValueError("Anthropic provider not initialized")
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            
            if not response.content or not response.content[0].text:
                raise ValueError("Invalid response format from Anthropic API: missing content")
            
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error generating text with Anthropic: {str(e)}")
            raise
    
    def generate_json(self, prompt: str, user_message: str) -> Dict[str, Any]:
        """
        Generate and parse JSON using the Anthropic API.
        
        Args:
            prompt: The system prompt
            user_message: The user message
            
        Returns:
            Dict[str, Any]: The parsed JSON response
        """
        import json
        import re
        
        if not self.client:
            if not self.initialize():
                raise ValueError("Anthropic provider not initialized")
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=prompt,
                messages=[{"role": "user", "content": user_message}]
            )
            
            if not response.content or not response.content[0].text:
                raise ValueError("Invalid response format from Anthropic API: missing content")
            
            content = response.content[0].text
            
            # Try to parse the response as JSON
            try:
                parsed_response = json.loads(content)
                return parsed_response
            except json.JSONDecodeError:
                # If the response is not valid JSON, try to extract JSON from the text
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    extracted_json = json_match.group(1)
                    parsed_response = json.loads(extracted_json)
                    return parsed_response
                else:
                    raise ValueError(f"Could not parse JSON from response: {content[:100]}...")
        except Exception as e:
            logger.error(f"Error generating JSON with Anthropic: {str(e)}")
            raise

class GroqProvider(LLMProvider):
    """Implementation of LLMProvider for Groq's API."""

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = os.getenv("GROQ_MODEL", "llama3-8b-8192")
        self.client = None

    def initialize(self) -> bool:
        """Initialize the Groq provider with the API key."""
        try:
            from groq import Groq

            if not self.api_key:
                logger.error("Cannot initialize Groq: API key not configured")
                return False

            self.client = Groq(api_key=self.api_key)

            logger.info("Groq provider initialized successfully")
            logger.info(f"Using Groq model: {self.model}")
            return True
        except ImportError:
            logger.error("Failed to import groq. Make sure it's installed.")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Groq provider: {str(e)}")
            return False

    def generate_text(self, prompt: str) -> str:
        """Generate text using the Groq API."""
        if not self.client:
            if not self.initialize():
                raise ValueError("Groq provider not initialized")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )

            if not response.choices or not response.choices[0].message.content:
                raise ValueError("Invalid response format from Groq API: missing content")

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating text with Groq: {str(e)}")
            raise

    def generate_json(self, prompt: str, user_message: str) -> Dict[str, Any]:
        """Generate and parse JSON using the Groq API."""
        import json
        import re

        if not self.client:
            if not self.initialize():
                raise ValueError("Groq provider not initialized")

        # Add explicit JSON instructions to ensure proper formatting
        json_system_prompt = f"{prompt}\n\nIMPORTANT: Your response MUST be valid JSON format ONLY, without any explanations or text outside the JSON structure. The response MUST be a valid JSON object that follows the schema described in the prompt. Do not include markdown formatting like ```json or ``` around your response."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": json_system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=4000,  # Ensure we get a complete response
            )

            if not response.choices or not response.choices[0].message.content:
                raise ValueError("Invalid response format from Groq API: missing content")

            content = response.choices[0].message.content
            
            # Try to extract JSON if it appears to be wrapped in markdown code blocks or has preamble text
            try:
                # First, try direct parsing
                return json.loads(content)
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract JSON from the response
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    extracted_json = json_match.group(1)
                    return json.loads(extracted_json)
                
                # If that fails too, look for anything that looks like a JSON object
                json_match = re.search(r'\{\s*"[^"]+"\s*:', content)
                if json_match:
                    start_idx = json_match.start()
                    # Try to find the complete JSON object
                    brace_count = 0
                    in_string = False
                    escape_next = False
                    for i in range(start_idx, len(content)):
                        char = content[i]
                        if escape_next:
                            escape_next = False
                            continue
                        if char == '\\"' and not escape_next and in_string:
                            escape_next = True
                        elif char == '"' and not escape_next:
                            in_string = not in_string
                        elif not in_string:
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    extracted_json = content[start_idx:i+1]
                                    return json.loads(extracted_json)
                
                # If all extraction methods fail, raise the original error
                raise ValueError(f"Could not parse JSON from response: {content[:100]}...")
        except Exception as e:
            logger.error(f"Error generating JSON with Groq: {str(e)}")
            logger.error(f"Raw response content: {content[:500] if 'content' in locals() else 'No content'}")
            raise

# Dictionary mapping provider names to their classes
PROVIDER_CLASSES: Dict[str, Type[LLMProvider]] = {
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "groq": GroqProvider,
}

def get_provider(provider_name: Optional[str] = None) -> LLMProvider:
    """
    Get an instance of the specified LLM provider.
    
    Args:
        provider_name: The name of the provider to use. If None, uses the provider
                      specified in the LLM_PROVIDER environment variable, or falls
                      back to "gemini" if not specified.
    
    Returns:
        LLMProvider: An instance of the specified provider
    
    Raises:
        ValueError: If the specified provider is not supported
    """
    if provider_name is None:
        provider_name = os.getenv("LLM_PROVIDER", "gemini").lower()
    
    if provider_name not in PROVIDER_CLASSES:
        supported_providers = ", ".join(PROVIDER_CLASSES.keys())
        raise ValueError(f"Unsupported LLM provider: {provider_name}. Supported providers: {supported_providers}")
    
    provider_class = PROVIDER_CLASSES[provider_name]
    provider = provider_class()
    
    # Initialize the provider
    if not provider.initialize():
        logger.warning(f"Failed to initialize {provider_name} provider. Some functionality may not work.")
    
    return provider

def get_available_providers() -> List[str]:
    """
    Get a list of available LLM providers.
    
    Returns:
        List[str]: A list of available provider names
    """
    return list(PROVIDER_CLASSES.keys())