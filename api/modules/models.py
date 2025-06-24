from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from enum import Enum

class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None
    message_history: Optional[List[Dict[str, str]]] = None
    test_mode: bool = False

class LLMReply(BaseModel):
    scheduling_problem: Dict[str, Any]
    clarification_question: Optional[str]
    is_complete: bool
    ready_to_solve: bool = False
    requires_support: bool = False

class ChatResponse(BaseModel):
    response: str
    requires_support: bool = False
    scheduling_problem: Dict[str, Any] = {}
    is_problem_complete: bool = False
    api_response: Optional[Dict[str, Any]] = None

class Job(BaseModel):
    job_id: int
    rig_id: int
    processing_time: int = 1

class SchedulingProblem(BaseModel):
    machines: List[Dict[str, int]]
    jobs: List[Job]
    rig_change_times: List[List[int]]
    solver_settings: Dict[str, Any]
    timeunit: Optional[str] = "minutes"

# OWPy API Models based on OpenAPI spec
class SolverFunction(str, Enum):
    GLOBAL = "GLOBAL"
    TWO_STEP = "TWO_STEP"
    SA = "SA"
    BRANCH_AND_BOUND = "BranchAndBound"

class SolverSettings(BaseModel):
    MaxTime: int = Field(..., description="Maximum time in seconds allowed to run the solver")
    UseHeuristics: bool = Field(..., description="Whether to use heuristic to get faster results")

class IOScenario(BaseModel):
    Machines: List[Dict[str, int]] = Field(..., description="The list of machines available for the current shift")
    Jobs: List[Dict[str, int]] = Field(..., description="The list of jobs to be distributed and processed")
    Rigs: List[List[int]] = Field(..., description="The estimated time for changing a machine from one rig to another")

class ScenarioOptimizeRequest(BaseModel):
    scenarioInput: IOScenario
    solverFunc: SolverFunction
    solver_settings: SolverSettings

class SolutionAndTime(BaseModel):
    makespan: int = Field(..., description="Maximum time the machines have for job processing and rig changes")
    machines_distribution: Dict[str, List[int]] = Field(..., description="Processing sequence of jobs per machine")
    time_needed_in_s: int = Field(..., description="Time needed to solve the problem in seconds")
