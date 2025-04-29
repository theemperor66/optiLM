from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Import modules
from .modules.models import ChatRequest, ChatResponse
from .modules.utils import analyze_message, generate_response, generate_general_response
from .modules.problem import formulate_scheduling_problem
from .modules.owpy import call_owpy_api

# Create FastAPI app
app = FastAPI(
    title="OWPy Chatbot API",
    description="A chatbot API that helps users with OWPy optimization library",
    version="1.0.0"
)

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
