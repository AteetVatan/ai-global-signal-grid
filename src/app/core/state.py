"""
State management models for Global Signal Grid (MASX) Agentic AI System.

Defines the core state objects for orchestrating multi-agent workflows, tracking agent execution,
and maintaining workflow progress and results. All models are Pydantic-based for type safety.

Usage: from app.core.state import MASXState, AgentState, WorkflowState
These models are used throughout the orchestrator, agents, and logging/auditing subsystems.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """
    State for an agent in the workflow.
    Tracks status, input/output, errors, and timing.
    """

    name: str = Field(..., description="Agent name")
    status: str = Field(
        "pending", description="Status: pending, running, success, failed"
    )
    input: Optional[Dict[str, Any]] = Field(
        default=None, description="Input data for the agent"
    )
    output: Optional[Dict[str, Any]] = Field(
        default=None, description="Output/result from the agent"
    )
    error: Optional[str] = Field(default=None, description="Error message if failed")
    start_time: Optional[datetime] = Field(
        default=None, description="Agent execution start time"
    )
    end_time: Optional[datetime] = Field(
        default=None, description="Agent execution end time"
    )
    logs: List[Dict[str, Any]] = Field(
        default_factory=list, description="Structured logs for this agent"
    )


class WorkflowState(BaseModel):
    """
    Tracks workflow progress, steps, and results.
    """

    current_step: Optional[str] = Field(default=None, description="Current step name")
    steps: List[str] = Field(
        default_factory=list, description="Ordered list of workflow steps"
    )
    completed: bool = Field(
        default=False, description="Whether the workflow is complete"
    )
    failed: bool = Field(default=False, description="Whether the workflow failed")
    results: Optional[Dict[str, Any]] = Field(
        default=None, description="Final results of the workflow"
    )


class MASXState(BaseModel):
    """
    Top-level state for a workflow run.
    Includes run metadata, agent states, workflow state, and errors.
    """

    run_id: str = Field(..., description="Unique identifier for this workflow run")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Run start timestamp (UTC)"
    )
    agents: Dict[str, AgentState] = Field(
        default_factory=dict, description="States for each agent by name"
    )
    workflow: WorkflowState = Field(..., description="Workflow progress and results")
    errors: List[str] = Field(
        default_factory=list, description="List of errors encountered during run"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional run metadata (config, env, etc.)"
    )
