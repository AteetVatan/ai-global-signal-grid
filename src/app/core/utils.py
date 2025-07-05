"""
Utility for Global Signal Grid (MASX) Agentic AI System.
Common utilities used across agents, workflows, and services including:
- ID generation, text sanitization, URL validation
- Retry logic with exponential backoff
- Performance measurement and timing utilities
Usage: from app.core.utils import generate_run_id, retry_with_backoff, measure_execution_time
"""

import time
import uuid
import re
import httpx
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Optional
from urllib.parse import urlparse

from .exceptions import ExternalServiceException


def generate_run_id() -> str:
    """
    Generate a unique run ID for workflow execution.
    Returns:
        str: Unique run ID in format 'YYYY-MM-DD-HHMMSS-UUID'
    """
    timestamp = time.strftime("%Y-%m-%d-%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{timestamp}-{unique_id}"


def sanitize_text(text: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize text input by removing control characters and limiting length.
    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length (None for no limit)
    Returns:
        str: Sanitized text
    """
    if not text:
        return ""

    # Remove control characters except newlines and tabs
    sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

    # Normalize whitespace
    sanitized = re.sub(r"\s+", " ", sanitized).strip()

    # Limit length if specified
    if max_length and len(sanitized) > max_length:
        sanitized = sanitized[: max_length - 3] + "..."

    return sanitized


def validate_url(url: str, timeout: int = 10) -> bool:
    """
    Validate URL format and optionally check accessibility.
    Args:
        url: URL to validate
        timeout: Timeout for accessibility check in seconds
    Returns:
        bool: True if URL is valid and accessible
    """
    try:
        # Basic URL format validation
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            return False

        # Check if URL is accessible
        with httpx.Client(timeout=timeout) as client:
            response = client.head(url, follow_redirects=True)
            return response.status_code < 400

    except Exception:
        return False


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    Decorator for retrying functions with exponential backoff.
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff calculation
        exceptions: Tuple of exceptions to catch and retry
    Usage:
        @retry_with_backoff(max_attempts=3, base_delay=1.0)
        def api_call():
            # Function that may fail
            pass
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts - 1:
                        # Last attempt, re-raise the exception
                        raise ExternalServiceException(
                            f"Function {func.__name__} failed after {max_attempts} attempts",
                            context={"last_error": str(e)},
                        )

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base**attempt), max_delay)
                    time.sleep(delay)

            # This should never be reached, but just in case
            raise last_exception

        return wrapper

    return decorator


# @contextmanager --Python decorator that lets you write your own with blocks.
@contextmanager
def measure_execution_time(operation_name: str = "operation"):
    """
    Context manager to measure execution time of operations.
    Args:
        operation_name: Name of the operation being measured
    Usage:
        with measure_execution_time("api_call"):
            # Code to measure
            result = api_call()
    """
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        duration = end_time - start_time
        print(f"{operation_name} took {duration:.2f} seconds")


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Safely parse JSON string with fallback to default value.
    Args:
        json_str: JSON string to parse
        default: Default value if parsing fails
    Returns:
        Parsed JSON object or default value
    """
    import json

    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def chunk_list(lst: list, chunk_size: int) -> list:
    """
    Split a list into chunks of specified size.
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
    Returns:
        List of chunks
    """
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]
