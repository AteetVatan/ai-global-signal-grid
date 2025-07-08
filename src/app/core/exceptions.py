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


class DatabaseException(MASXException):
    """Exception raised for database-related errors."""

    pass


class ConfigurationException(MASXException):
    """Exception raised for configuration-related errors."""

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


class TranslationException(MASXException):
    """Exception raised for translation-related errors."""

    pass


class EmbeddingException(MASXException):
    """Exception raised for embedding-related errors."""

    pass


class DataSourceError(MASXException):
    """Exception raised for data source-related errors."""

    pass


class ValidationError(MASXException):
    """Exception raised for validation-related errors."""

    pass


class ProcessingError(MASXException):
    """Exception raised for data processing-related errors."""

    pass


class StreamingError(MASXException):
    """Exception raised for streaming-related errors."""

    pass


class AuthenticationError(MASXException):
    """Exception raised for authentication-related errors."""

    pass


class AnalyticsError(MASXException):
    """Exception raised for analytics-related errors."""

    pass
