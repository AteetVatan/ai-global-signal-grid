"""
Global Signal Grid (GSG)
A modular, multi-agent system for geopolitical intelligence gathering.
Powered by LangGraph, CrewAI, and AutoGen for collaborative reasoning.

Author: MASX AI Team
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "MASX AI Team"
__email__ = "ab@masxai.com"

# Import core components for easy access
from .config.settings import get_settings
from .core.state import MASXState
from .workflows.orchestrator import MASXOrchestrator

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "get_settings",
    "MASXState",
    "MASXOrchestrator",
]
