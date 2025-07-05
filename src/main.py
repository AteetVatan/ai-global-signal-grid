"""
Main application entry point for Global Signal Grid (MASX) Agentic AI System.

This module provides the FastAPI application server with:
- Health check endpoints
- Workflow management
- Data retrieval and analytics
- Service monitoring
- Interactive API documentation

Usage:
    python -m src.main
    uvicorn src.main:app --host 0.0.0.0 --port 8000
"""

import uvicorn
from app.api import create_app
from app.config.settings import get_settings

# Create FastAPI application
app = create_app()

if __name__ == "__main__":
    settings = get_settings()
    
    # Run the application
    uvicorn.run(
        "src.main:app",
        host=settings.api_host or "0.0.0.0",
        port=settings.api_port or 8000,
        reload=settings.api_reload or False,
        log_level=settings.log_level.lower() if settings.log_level else "info"
    ) 