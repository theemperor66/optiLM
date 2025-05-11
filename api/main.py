from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
import logging
import time
import traceback
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("api")

# Load environment variables from .env file
load_dotenv()

# Import modules
from .modules.models import ChatRequest, ChatResponse, SolverFunction, SchedulingProblem
from .modules.utils import analyze_message, generate_response, generate_general_response
from .modules.problem import formulate_scheduling_problem, interactive_step
from .modules.owpy import call_owpy_api

# Create FastAPI app
app = FastAPI(
    title="OWPy Chatbot API",
    description="A chatbot API that helps users with OWPy optimization library",
    version="1.0.0"
)

# Add middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    logger.info(f"Request {request_id} started: {request.method} {request.url.path}")
    start_time = time.time()

    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"Request {request_id} completed: {response.status_code} in {process_time:.3f}s")
        return response
    except Exception as e:
        logger.error(f"Request {request_id} failed: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {str(e)}"}
        )

@app.get("/")
async def root():
    logger.info("Health check endpoint called")
    return {"message": "OWPy Chatbot API", "status": "running"}

@app.get("/hello/{name}")
async def say_hello(name: str):
    logger.info(f"Hello endpoint called with name: {name}")
    return {"message": f"Hello {name}"}

@app.get("/api/v1/solvers/")
async def available_solvers():
    """
    Return the list of available solvers.

    This endpoint is used by the UI to populate the solver selection dropdown.
    """
    logger.info("Available solvers endpoint called")
    # Get the list of available solvers from the SolverFunction enum
    solvers = [solver.value for solver in SolverFunction]
    logger.debug(f"Available solvers: {solvers}")
    return solvers

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a chat message from the user and provide a response.

    The chatbot will:
    1. Process the user's message in the context of the current problem state
    2. Ask for missing information if needed
    3. Make API calls to OWPy when the problem is complete and the user asks to solve it
    4. Refer to support team if the query is too complex
    """
    request_id = str(uuid.uuid4())
    user_message = request.message
    state_in = request.context or {}
    test_mode = request.test_mode

    logger.info(f"Chat request {request_id} received: {user_message[:50]}{'...' if len(user_message) > 50 else ''}")
    logger.debug(f"Full message: {user_message}")
    logger.debug(f"Context: {state_in}")
    logger.debug(f"Test mode: {test_mode}")

    # Check if Google API key is set
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("Google API key not configured")
        return ChatResponse(
            response="API key not configured. Please set the GOOGLE_API_KEY environment variable.",
            requires_support=True
        )

    # Quick reset
    if user_message.strip().lower() in {"reset", "start over", "new problem"}:
        logger.info(f"User requested reset for request {request_id}")
        return ChatResponse(
            response="Got it – starting a fresh problem. Tell me about your machines.",
            scheduling_problem={}
        )

    # Check if the user wants to solve the problem
    confirm = any(w in user_message.lower() for w in ["solve", "optimize", "finish"])

    # Process the user message with the interactive builder
    try:
        llm = interactive_step(user_message, state_in, confirm, test_mode)
        logger.info(f"Interactive step completed for request {request_id}, is_complete={llm.is_complete}, ready_to_solve={llm.ready_to_solve}")
    except Exception as e:
        logger.error(f"Error in interactive step for request {request_id}: {str(e)}")
        logger.error(traceback.format_exc())
        return ChatResponse(
            response=f"I'm having trouble processing your message. Error: {str(e)}",
            requires_support=True
        )

    # Set the assistant's response text
    assistant_text = llm.clarification_question or "Great, I have the full problem."

    # If the problem is complete and the user wants to solve it, call OWPy
    owpy_result = None
    if llm.ready_to_solve:
        try:
            logger.info(f"Calling OWPy API for request {request_id} (test_mode={test_mode})")
            owpy_result = call_owpy_api(SchedulingProblem(**llm.scheduling_problem), test_mode)

            if owpy_result.get("status") == "error":
                logger.error(f"OWPy API error for request {request_id}: {owpy_result.get('error', 'Unknown error')}")
                return ChatResponse(
                    response=generate_response(user_message, owpy_result),
                    scheduling_problem=llm.scheduling_problem,
                    is_problem_complete=False,
                    api_response=owpy_result,
                    requires_support=True
                )

            logger.info(f"OWPy API call successful for request {request_id}")
            assistant_text = generate_response(user_message, owpy_result)
        except Exception as e:
            logger.error(f"Error calling OWPy API for request {request_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return ChatResponse(
                response=f"An error occurred while solving your problem: {str(e)}",
                scheduling_problem=llm.scheduling_problem,
                is_problem_complete=False,
                requires_support=True
            )

    # Return the response
    return ChatResponse(
        response=assistant_text,
        scheduling_problem=llm.scheduling_problem,
        is_problem_complete=llm.ready_to_solve and owpy_result and owpy_result.get("status") == "success",
        api_response=owpy_result,
        requires_support=llm.requires_support
    )
