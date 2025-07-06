"""
Configuration management for MASX Agentic AI System.

This package handles all configuration settings, environment variables,
and system-wide constants used throughout the application.
"""

from .settings import get_settings, Settings # initialize the sttings for e.g. enviorment variables
from .logging_config import setup_logging

__all__ = ["get_settings", "Settings", "setup_logging"]
