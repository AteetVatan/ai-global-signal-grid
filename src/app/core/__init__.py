# ┌───────────────────────────────────────────────────────────────┐
# │  Copyright (c) 2025 Ateet Vatan Bahmani                      │
# │  Project: MASX AI – Strategic Agentic AI System              │
# │  All rights reserved.                                        │
# └───────────────────────────────────────────────────────────────┘
#
# MASX AI is a proprietary software system developed and owned by Ateet Vatan Bahmani.
# The source code, documentation, workflows, designs, and naming (including "MASX AI")
# are protected by applicable copyright and trademark laws.
#
# Redistribution, modification, commercial use, or publication of any portion of this
# project without explicit written consent is strictly prohibited.
#
# This project is not open-source and is intended solely for internal, research,
# or demonstration use by the author.
#
# Contact: ab@masxai.com | MASXAI.com

"""
Package contains ---->>> state management,
exception handling, and utility functions used throughout the application.
"""

from .state import MASXState, AgentState, WorkflowState
from .language_utils import LanguageUtils
from .querystate import QueryState, FeedEntry, QueryTranslated
from .flashpoint import FlashpointItem, FlashpointDataset
from .flashpointstore import FlashpointStore
from .exceptions import (
    MASXException,
    AgentException,
    WorkflowException,
    ConfigurationException,
    ValidationException,
    ExternalServiceException,
)
from .utils import (
    generate_workflow_id,
    sanitize_text,
    validate_url,
    retry_with_backoff,
    measure_execution_time, 
)

__all__ = [
    "MASXState",
    "AgentState",
    "WorkflowState",
    "MASXException",
    "AgentException",
    "WorkflowException",
    "ConfigurationException",
    "ValidationException",
    "ExternalServiceException",
    "generate_workflow_id",
    "sanitize_text",
    "validate_url",
    "retry_with_backoff",
    "measure_execution_time",
    "LanguageUtils",
    "QueryState",
    "FeedEntry",
    "QueryTranslated",
    "FlashpointItem",
    "FlashpointDataset",
    "FlashpointStore",
]
