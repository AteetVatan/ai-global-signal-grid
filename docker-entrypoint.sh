#!/bin/bash

# Global Signal Grid (MASX) Agentic AI System Docker Entrypoint
# This script can start either the FastAPI server, the daily job scheduler, or both

set -e

# Function to start FastAPI server
start_fastapi() {
    echo "🚀 Starting FastAPI server..."
    cd /app
    python main_fast_api.py
}

# Function to start daily job scheduler
start_scheduler() {
    echo "⏰ Starting daily job scheduler..."
    cd /app
    python daily_job.py
}

# Function to start both services (FastAPI in background, scheduler in foreground)
start_both() {
    echo "🔄 Starting both FastAPI server and daily job scheduler..."
    cd /app
    
    # Start FastAPI in background
    python main_fast_api.py &
    FASTAPI_PID=$!
    
    # Wait a moment for FastAPI to start
    sleep 5
    
    # Start scheduler in foreground
    python daily_job.py &
    SCHEDULER_PID=$!
    
    # Function to handle shutdown
    shutdown() {
        echo "🛑 Shutting down services..."
        kill $FASTAPI_PID 2>/dev/null || true
        kill $SCHEDULER_PID 2>/dev/null || true
        exit 0
    }
    
    # Set up signal handlers
    trap shutdown SIGTERM SIGINT
    
    # Wait for either process to exit
    wait $FASTAPI_PID $SCHEDULER_PID
}

# Main logic based on environment variables
case "${SERVICE_TYPE:-both}" in
    "fastapi")
        start_fastapi
        ;;
    "scheduler")
        start_scheduler
        ;;
    "both"|"")
        start_both
        ;;
    *)
        echo "❌ Invalid SERVICE_TYPE: ${SERVICE_TYPE}"
        echo "Valid options: fastapi, scheduler, both"
        exit 1
        ;;
esac 