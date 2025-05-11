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
from .modules.models import ChatRequest, ChatResponse, SolverFunction
from .modules.utils import analyze_message, generate_response, generate_general_response
from .modules.problem import formulate_scheduling_problem
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
    1. Analyze the user's message
    2. Attempt to formulate an optimization problem if applicable
    3. Make API calls to OWPy if a valid problem is formulated
    4. Refer to support team if the query is too complex
    """
    request_id = str(uuid.uuid4())
    user_message = request.message
    context = request.context or {}
    test_mode = request.test_mode

    logger.info(f"Chat request {request_id} received: {user_message[:50]}{'...' if len(user_message) > 50 else ''}")
    logger.debug(f"Full message: {user_message}")
    logger.debug(f"Context: {context}")
    logger.debug(f"Test mode: {test_mode}")

    # Check if Google API key is set
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("Google API key not configured")
        return ChatResponse(
            response="API key not configured. Please set the GOOGLE_API_KEY environment variable.",
            requires_support=True
        )

    # Process the user message to understand the intent
    try:
        intent, complexity = analyze_message(user_message)
        logger.info(f"Message analysis for {request_id}: intent={intent}, complexity={complexity}")
    except Exception as e:
        logger.error(f"Error analyzing message: {str(e)}")
        return ChatResponse(
            response=f"I'm having trouble understanding your message. Error: {str(e)}",
            requires_support=True
        )

    # If the query is too complex, refer to support
    if complexity > 7:  # On a scale of 1-10
        logger.info(f"Query {request_id} too complex (score: {complexity}), referring to support")
        return ChatResponse(
            response="Your query appears to be complex. I recommend reaching out to our support team for assistance.",
            requires_support=True
        )

    # If the intent is to solve a scheduling problem
    if intent == "solve_optimization":
        logger.info(f"Processing optimization problem for request {request_id}")
        try:
            # Formulate the scheduling problem
            logger.debug(f"Formulating scheduling problem for request {request_id}")
            problem = formulate_scheduling_problem(user_message, context, test_mode)
            logger.info(f"Problem formulated successfully for request {request_id}")

            # Call OWPy API or generate random solution in test mode
            logger.debug(f"Calling OWPy API for request {request_id} (test_mode={test_mode})")
            api_response = call_owpy_api(problem, test_mode)

            if api_response.get("status") == "error":
                logger.error(f"OWPy API error for request {request_id}: {api_response.get('error', 'Unknown error')}")
                return ChatResponse(
                    response=generate_response(user_message, api_response),
                    scheduling_problem=problem.dict(),
                    api_response=api_response,
                    requires_support=True
                )

            logger.info(f"OWPy API call successful for request {request_id}")
            return ChatResponse(
                response=generate_response(user_message, api_response),
                scheduling_problem=problem.dict(),
                api_response=api_response
            )
        except ValueError as e:
            logger.error(f"Value error formulating problem for request {request_id}: {str(e)}")
            return ChatResponse(
                response=f"I couldn't formulate your scheduling problem: {str(e)}",
                requires_support=True
            )
        except Exception as e:
            logger.error(f"Unexpected error processing request {request_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return ChatResponse(
                response=f"An error occurred while processing your request: {str(e)}. Please try again with a clearer description of your scheduling problem.",
                requires_support=True
            )

    # For other intents, provide a general response
    logger.info(f"Generating general response for request {request_id} with intent {intent}")
    return ChatResponse(
        response=generate_general_response(user_message, intent)
    )
