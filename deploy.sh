#!/bin/bash

# MASX Global Signal Grid Deployment Script
# This script provides easy deployment commands for different scenarios

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    if [ ! -f ".env" ]; then
        print_warning ".env file not found. Creating from env.example..."
        if [ -f "env.example" ]; then
            cp env.example .env
            print_status "Please edit .env file with your configuration before continuing."
        else
            print_error "env.example file not found. Please create a .env file with your configuration."
            exit 1
        fi
    fi
    
    print_status "Prerequisites check completed."
}

# Function to build Docker image
build_image() {
    print_status "Building Docker image..."
    docker-compose build
    print_status "Docker image built successfully."
}

# Function to start services
start_services() {
    local service_type=$1
    
    case $service_type in
        "fastapi")
            print_header "Starting FastAPI Server Only"
            docker-compose up -d masx-fastapi
            ;;
        "scheduler")
            print_header "Starting Daily Job Scheduler Only"
            docker-compose up -d masx-scheduler
            ;;
        "both"|"full")
            print_header "Starting Both Services (FastAPI + Scheduler)"
            docker-compose up -d masx-full
            ;;
        "dev")
            print_header "Starting Development Mode"
            docker-compose up -d masx-dev
            ;;
        *)
            print_error "Invalid service type: $service_type"
            print_status "Valid options: fastapi, scheduler, both, dev"
            exit 1
            ;;
    esac
    
    print_status "Services started successfully."
}

# Function to stop services
stop_services() {
    print_header "Stopping All Services"
    docker-compose down
    print_status "All services stopped."
}

# Function to restart services
restart_services() {
    local service_type=$1
    
    print_header "Restarting Services"
    docker-compose down
    start_services $service_type
}

# Function to show logs
show_logs() {
    local service_name=$1
    
    if [ -z "$service_name" ]; then
        print_header "Showing All Logs"
        docker-compose logs -f
    else
        print_header "Showing Logs for $service_name"
        docker-compose logs -f $service_name
    fi
}

# Function to show status
show_status() {
    print_header "Service Status"
    docker-compose ps
}

# Function to clean up
cleanup() {
    print_header "Cleaning Up"
    docker-compose down -v
    docker system prune -f
    print_status "Cleanup completed."
}

# Function to show help
show_help() {
    echo "MASX Global Signal Grid Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  build                    Build Docker image"
    echo "  start [TYPE]             Start services"
    echo "  stop                     Stop all services"
    echo "  restart [TYPE]           Restart services"
    echo "  logs [SERVICE]           Show logs"
    echo "  status                   Show service status"
    echo "  cleanup                  Clean up containers and volumes"
    echo "  help                     Show this help message"
    echo ""
    echo "Service Types:"
    echo "  fastapi                  FastAPI server only"
    echo "  scheduler                Daily job scheduler only"
    echo "  both|full                Both services (default)"
    echo "  dev                      Development mode with hot reload"
    echo ""
    echo "Examples:"
    echo "  $0 start both            Start both services"
    echo "  $0 start fastapi         Start FastAPI server only"
    echo "  $0 logs masx-full        Show logs for full service"
    echo "  $0 restart dev           Restart in development mode"
}

# Main script logic
case "${1:-help}" in
    "build")
        check_prerequisites
        build_image
        ;;
    "start")
        check_prerequisites
        start_services "${2:-both}"
        ;;
    "stop")
        stop_services
        ;;
    "restart")
        check_prerequisites
        restart_services "${2:-both}"
        ;;
    "logs")
        show_logs "$2"
        ;;
    "status")
        show_status
        ;;
    "cleanup")
        cleanup
        ;;
    "help"|"--help"|"-h")
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac 