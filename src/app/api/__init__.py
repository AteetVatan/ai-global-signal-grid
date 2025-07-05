"""
FastAPI API layer for Global Signal Grid (MASX) Agentic AI System.

Provides REST API endpoints for:
- System monitoring and health checks
- Workflow management and execution
- Data retrieval and analysis
- Service status and configuration

Usage:
    from app.api import create_app
    
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

from .app import create_app
from .routes import health, workflows, data, services

__all__ = [
    "create_app",
    "health",
    "workflows", 
    "data",
    "services",
] 