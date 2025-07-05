"""
Exception for Global Signal Grid (MASX) Agentic AI System.
Defines custom exceptions for robust error handling across agents, workflows, configuration,
validation, and external service integration. Use these for clear, structured error reporting.

Usage: from app.core.exceptions import MASXException, AgentException, WorkflowException

All exceptions inherit from MASXException and can include a message and optional context.
"""

from typing import Any, Optional


class MASXException(Exception):
    """
    Base exception for all MASX/Global Signal Grid errors.
    """

    def __init__(self, message: str, context: Optional[Any] = None):
        super().__init__(message)
        self.context = context


class AgentException(MASXException):
    """
    Exception raised for agent-specific errors (logic, execution, etc.).
    """

    pass


class WorkflowException(MASXException):
    """
    Exception raised for workflow orchestration errors.
    """

    pass


class ConfigurationException(MASXException):
    """
    Exception raised for configuration or environment issues.
    """

    pass


class ValidationException(MASXException):
    """
    Exception raised for schema or data validation errors.
    """

    pass


class ExternalServiceException(MASXException):
    """
    Exception raised for external API/network/service failures.
    """

    pass
