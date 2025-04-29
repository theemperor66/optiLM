# OWPy Chatbot for Scheduling Problems

## Project Overview

This project implements a chatbot system that helps users interact with the OWPy scheduling library. The chatbot can understand natural language queries about scheduling problems, formulate them into structured scheduling problems, and solve them using the OWPy API. The project includes both a backend API and a user-friendly Streamlit interface for building and visualizing scheduling problems.

## Features

- **Natural Language Processing**: Understands user queries about scheduling problems
- **Problem Formulation**: Converts natural language descriptions into structured scheduling problems
- **API Integration**: Communicates with the OWPy API to solve scheduling problems
- **Test Mode**: Generates random solutions without calling the OWPy API for testing and demonstration purposes
- **User-Friendly Responses**: Provides human-readable responses explaining the solutions
- **Support Referral**: Identifies complex queries and refers users to the support team when necessary
- **Interactive UI**: Streamlit interface for building and visualizing scheduling problems
- **Visualization**: Graphical representation of problems and solutions, including Gantt charts for scheduling
- **Problem Builder**: Form-based interface for defining machines, jobs, and rig requirements

## Architecture

The system consists of two main components:

### Backend API (FastAPI)

The backend is built using FastAPI and consists of the following components:

1. **Chat Endpoint**: Receives user messages and processes them
2. **Message Analyzer**: Determines the intent and complexity of user queries
3. **Problem Formulator**: Converts natural language into structured scheduling problems
4. **API Client**: Communicates with the OWPy API to solve scheduling problems
5. **Response Generator**: Creates user-friendly responses based on API results

### Frontend UI (Streamlit)

The frontend is built using Streamlit and consists of the following components:

1. **Scheduling Problem Builder**: Form-based interface for defining machines, jobs, and rig requirements
2. **Chat Interface**: Natural language interface for describing scheduling problems
3. **Problem Visualizer**: Displays the machines, jobs, and rig change times
4. **Solution Visualizer**: Displays the solution as a Gantt chart showing job assignments to machines
5. **API Connector**: Communicates with the FastAPI backend

## Directory Structure

The project is organized into the following directory structure:

```
optimLM/
├── api/                    # Backend API code
│   ├── modules/            # Backend modules
│   │   ├── models.py       # Data models
│   │   ├── llm.py          # LLM integration
│   │   ├── owpy.py         # OWPy API integration
│   │   ├── problem.py      # Problem formulation
│   │   └── utils.py        # Utility functions
│   └── main.py             # FastAPI application
├── ui/                     # Frontend UI code
│   ├── modules/            # Frontend modules
│   │   ├── api_client.py   # API client
│   │   └── visualization.py # Visualization functions
│   ├── pages/              # UI pages
│   │   ├── problem_builder.py # Problem Builder page
│   │   └── chat_interface.py # Chat Interface page
│   └── streamlit_app.py    # Streamlit application
├── run_api.py              # Script to run the API
├── run_ui.py               # Script to run the UI
├── system_prompt.txt       # System prompt for LLM
├── requirements.txt        # Project dependencies
├── Dockerfile.api          # Dockerfile for the API
├── Dockerfile.streamlit    # Dockerfile for the UI
└── docker-compose.yml      # Docker Compose configuration
```

## API Endpoints

### GET /

Returns a simple status message indicating that the API is running.

### GET /hello/{name}

A simple greeting endpoint that returns a hello message with the provided name.

### POST /chat

The main endpoint for chatbot interaction. It accepts a JSON payload with a user message and optional context, processes the message, and returns a response.

#### Request Format

```json
{
  "message": "Your scheduling problem description",
  "context": {
    "machines": [...],
    "jobs": [...],
    "rig_change_times": [...],
    "solver_settings": {...}
  },
  "test_mode": false
}
```

The `context` field is optional and can be used to provide pre-formulated parts of the scheduling problem.
The `test_mode` field is optional (defaults to false) and can be used to generate random solutions without calling the OWPy API.

#### Response Format

```json
{
  "response": "Human-readable response",
  "requires_support": false,
  "scheduling_problem": {
    "machines": [...],
    "jobs": [...],
    "rig_change_times": [...],
    "solver_settings": {...}
  },
  "api_response": {
    "status": "success",
    "solution": {...}
  }
}
```

- `response`: A human-readable response to the user's query
- `requires_support`: A boolean indicating whether the query is too complex and requires support
- `scheduling_problem`: The formulated scheduling problem (if applicable)
- `api_response`: The response from the OWPy API (if applicable)

## Usage Examples

### General Query

**Request:**
```json
{
  "message": "What can you help me with?"
}
```

**Response:**
```json
{
  "response": "I'm an OWPy assistant designed to help with scheduling problems. You can ask me to formulate and solve scheduling problems for you. For example, you could describe a problem with 'machines, jobs, and rig requirements' and I'll help you find the optimal schedule to minimize the makespan.",
  "requires_support": false
}
```

### Simple Scheduling Problem

**Request:**
```json
{
  "message": "I have 2 machines and 3 jobs. Machine 1 has a processing time of 2 and Machine 2 has a processing time of 3. Job 1 requires rig 1, Job 2 requires rig 2, and Job 3 requires rig 1. The rig change time from rig 1 to rig 2 is 2 units, and from rig 2 to rig 1 is 1 unit."
}
```

**Response:**
```json
{
  "response": "I've solved your scheduling problem! The optimal makespan is 7 time units.\n\nThe job assignments to machines are:\n- Machine 1: job_1, job_3\n- Machine 2: job_2\n\nThe solution was found in 0.5 seconds.",
  "requires_support": false,
  "scheduling_problem": {
    "machines": [
      {"machine_id": 1, "processing_time": 2},
      {"machine_id": 2, "processing_time": 3}
    ],
    "jobs": [
      {"job_id": 1, "rig_id": 1},
      {"job_id": 2, "rig_id": 2},
      {"job_id": 3, "rig_id": 1}
    ],
    "rig_change_times": [
      [0, 2],
      [1, 0]
    ],
    "solver_settings": {
      "max_time": 60,
      "use_heuristics": true,
      "solver_function": "GLOBAL"
    }
  },
  "api_response": {
    "status": "success",
    "solution": {
      "objective_value": 7,
      "variables": {
        "job_1": "1",
        "job_2": "2",
        "job_3": "1"
      },
      "solve_time": 0.5
    }
  }
}
```

## Setup and Installation

### Option 1: Using Docker (Recommended)

1. Clone the repository
2. Create a `.env` file with your API keys (see API Integrations section below)
3. Build and start the containers:
   ```
   docker-compose up -d
   ```
4. The API will be available at http://localhost:8000
5. The Streamlit UI will be available at http://localhost:8501

To stop the containers:
```
docker-compose down
```

### Option 2: Manual Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up your environment variables in the `.env` file (see API Integrations section below)

#### Running the Backend API

1. Start the FastAPI server: `python run_api.py`
2. The API will be available at http://127.0.0.1:8000

#### Running the Streamlit UI

1. Make sure the FastAPI backend is running
2. Start the Streamlit app: `python run_ui.py`
3. The UI will open in your browser at http://localhost:8501

**Note**: The Streamlit UI requires the FastAPI backend to be running, as it makes API calls to the backend for processing optimization problems.

## API Integrations

This project integrates with two APIs:

1. **Google's Gemini 2.5 Pro** - For parsing natural language descriptions of optimization problems
2. **OWPy API** - For solving the formulated optimization problems

### Setting Up Gemini API

1. Obtain a Google API key from the [Google AI Studio](https://ai.google.dev/)
2. Add your API key to the `.env` file:
   ```
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

### Setting Up OWPy API

1. Obtain an API key for the OWPy optimization service
2. Add the API URL and key to the `.env` file:
   ```
   OWPY_API_URL=https://api.optware.com/owpy
   OWPY_API_KEY=your_actual_owpy_api_key_here
   ```

### Installing Dependencies

Make sure you have installed the required dependencies:
```
pip install -r requirements.txt
```

### How It Works

1. **Natural Language Processing with Gemini**:
   - The system uses a carefully crafted prompt (in `system_prompt.txt`) to instruct Gemini on how to extract optimization problem components from natural language.
   - When a user sends a message describing an optimization problem, the system sends the message along with the system prompt to Gemini.
   - Gemini extracts the problem type, variables, constraints, and objective.
   - The system validates the response and creates a structured optimization problem.

2. **Problem Conversion**:
   - The generic optimization problem is converted to the specific format required by the OWPy API.
   - This involves mapping our variables, constraints, and objectives to the scheduling problem format (machines, jobs, and rigs) expected by OWPy.

3. **API Integration with OWPy**:
   - The formatted problem is sent to the OWPy API's `/api/v1/solve/` endpoint.
   - Authentication is handled using the API key provided in the `.env` file.
   - The API solves the problem and returns a solution with makespan and job assignments.

4. **Response Processing**:
   - The API response is converted back to our internal format.
   - A human-readable response is generated explaining the solution.
   - The solution is presented to the user with the objective value, variable values, and solve time.

### Fallback and Error Handling

#### Gemini API Fallback
If there's an issue with the Gemini API (e.g., API key not set, network error, invalid response), the system will fall back to:
1. Using any problem components provided in the context
2. Using default values for missing components

#### OWPy API Error Handling
The system includes robust error handling for the OWPy API:
1. Authentication errors are caught and reported to the user
2. Invalid request errors are processed and the specific validation issues are extracted
3. Network and server errors are handled gracefully
4. Timeouts and other operational issues are reported with clear error messages

In all error cases, the user is provided with a helpful message explaining the issue and suggesting next steps.

## Streamlit UI

The Streamlit UI provides a user-friendly interface for interacting with the scheduling system. It consists of two main pages:

### Scheduling Problem Builder

The Scheduling Problem Builder page allows you to define scheduling problems using a form-based interface:

1. **Machines**: Add, edit, and remove machines with their IDs and processing times
2. **Jobs**: Add, edit, and remove jobs with their IDs and rig requirements
3. **Rig Change Times**: Define the time required to change from one rig to another using a matrix
4. **Solver Settings**: Configure the solver with maximum time, heuristics option, and solver function
5. **Visualization**: See a visual representation of your scheduling problem
6. **Solve**: Submit the problem to the backend and view the solution

The Scheduling Problem Builder is ideal for users who want precise control over the problem formulation.

### Chat Interface

The Chat Interface page allows you to describe scheduling problems in natural language:

1. **Natural Language Input**: Describe your scheduling problem in plain English
2. **Problem Extraction**: The system extracts the machines, jobs, and rig requirements from your description
3. **Visualization**: See a visual representation of the extracted scheduling problem
4. **Solution**: View the solution as a Gantt chart showing job assignments to machines

The Chat Interface is ideal for users who prefer to describe their problems conversationally.

### Visualization Features

The UI includes several visualization features:

1. **Problem Structure**: Tables showing machines, jobs, and rig requirements
2. **Rig Change Time Matrix**: A heatmap visualization of the rig change times
3. **Solution Display**: Clear presentation of the job assignments to machines
4. **Gantt Charts**: A Gantt chart showing the job assignments and timing on each machine

### Test Mode

The system includes a test mode that allows you to test the UI and API without making actual calls to the OWPy API:

1. **Toggle in UI**: A checkbox in the sidebar allows you to enable or disable test mode
2. **Random Solutions**: When test mode is enabled, the system generates random solutions instead of calling the OWPy API
3. **LLM Integration**: Even in test mode, the system still uses the LLM to parse natural language descriptions into structured scheduling problems
4. **Visualization**: All visualization features work with the randomly generated solutions

Test mode is useful for:
- Demonstrating the system without an OWPy API key
- Testing the UI and visualization features
- Developing and debugging without consuming API credits
- Presenting the system in environments without internet access

To enable test mode:
1. In the Streamlit UI: Check the "Test Mode (Random Solutions)" checkbox in the sidebar
2. In API calls: Set the `test_mode` parameter to `true` in the request body

## Docker Configuration

The Docker setup includes the following components:

1. **FastAPI Backend**: Runs on port 8000 and provides the API endpoints
2. **Streamlit Frontend**: Runs on port 8501 and provides the user interface

### Environment Variables

When using Docker, you can configure the application using environment variables:

1. Create a `.env` file in the project root with the following variables:
   ```
   GOOGLE_API_KEY=your_actual_api_key_here
   OWPY_API_URL=https://api.optware.com/owpy
   OWPY_API_KEY=your_actual_owpy_api_key_here
   ```

2. The docker-compose.yml file will automatically use these variables.

### Docker Volumes (Optional)

If you want to persist data between container restarts, you can add volumes to the docker-compose.yml file:

```yaml
services:
  api:
    # ... existing configuration ...
    volumes:
      - ./data:/app/data
```

### Custom Port Configuration

If you need to use different ports, you can modify the docker-compose.yml file:

```yaml
services:
  api:
    # ... existing configuration ...
    ports:
      - "8080:8000"  # Map host port 8080 to container port 8000

  streamlit:
    # ... existing configuration ...
    ports:
      - "8888:8501"  # Map host port 8888 to container port 8501
    environment:
      - API_URL=http://api:8000  # This should always point to the container port
```

## Future Improvements

- Support for more complex scheduling problem features (precedence constraints, release dates, due dates)
- User authentication and session management
- History tracking for user interactions
- Interactive problem refinement through multi-turn conversations
- Fine-tuning the Gemini model specifically for scheduling problems
- Enhanced Gantt chart visualizations with rig change indicators
- Export/import functionality for scheduling problem definitions
- Support for additional OWPy solver functions and parameters
