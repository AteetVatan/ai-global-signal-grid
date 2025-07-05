"""
Workflow orchestration for Global Signal Grid (MASX) Agentic AI System.
This package contains LangGraph workflow definitions and orchestration logic including:
- Main orchestrator for coordinating agent execution
- Workflow definitions for different scenarios
- Parallel execution handling and state management
- Conditional logic and error recovery

Usage:    from app.workflows import MASXOrchestrator, create_daily_workflow
"""

from .orchestrator import MASXOrchestrator
from .workflows import (
    create_daily_workflow,
    create_detection_workflow,
    create_trigger_workflow,
)

__all__ = [
    "MASXOrchestrator",
    "create_daily_workflow",
    "create_detection_workflow",
    "create_trigger_workflow",
]
