# MASX Global Signal Grid Deployment Script (PowerShell)
# This script provides easy deployment commands for different scenarios

param(
    [Parameter(Position = 0)]
    [string]$Command = "help",
    
    [Parameter(Position = 1)]
    [string]$ServiceType = "both"
)

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Write-Header {
    param([string]$Message)
    Write-Host "================================" -ForegroundColor Blue
    Write-Host $Message -ForegroundColor Blue
    Write-Host "================================" -ForegroundColor Blue
}

# Function to check prerequisites
function Test-Prerequisites {
    Write-Status "Checking prerequisites..."
    
    # Check if Docker is installed
    try {
        docker --version | Out-Null
    }
    catch {
        Write-Error "Docker is not installed. Please install Docker first."
        exit 1
    }
    
    # Check if Docker Compose is installed
    try {
        docker-compose --version | Out-Null
    }
    catch {
        Write-Error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    }
    
    # Check for .env file
    if (-not (Test-Path ".env")) {
        Write-Warning ".env file not found. Creating from env.example..."
        if (Test-Path "env.example") {
            Copy-Item "env.example" ".env"
            Write-Status "Please edit .env file with your configuration before continuing."
        }
        else {
            Write-Error "env.example file not found. Please create a .env file with your configuration."
            exit 1
        }
    }
    
    Write-Status "Prerequisites check completed."
}

# Function to build Docker image
function Build-Image {
    Write-Status "Building Docker image..."
    docker-compose build
    Write-Status "Docker image built successfully."
}

# Function to start services
function Start-Services {
    param([string]$ServiceType)
    
    switch ($ServiceType) {
        "fastapi" {
            Write-Header "Starting FastAPI Server Only"
            docker-compose up -d masx-fastapi
        }
        "scheduler" {
            Write-Header "Starting Daily Job Scheduler Only"
            docker-compose up -d masx-scheduler
        }
        "both" {
            Write-Header "Starting Both Services (FastAPI + Scheduler)"
            docker-compose up -d masx-full
        }
        "full" {
            Write-Header "Starting Both Services (FastAPI + Scheduler)"
            docker-compose up -d masx-full
        }
        "dev" {
            Write-Header "Starting Development Mode"
            docker-compose up -d masx-dev
        }
        default {
            Write-Error "Invalid service type: $ServiceType"
            Write-Status "Valid options: fastapi, scheduler, both, dev"
            exit 1
        }
    }
    
    Write-Status "Services started successfully."
}

# Function to stop services
function Stop-Services {
    Write-Header "Stopping All Services"
    docker-compose down
    Write-Status "All services stopped."
}

# Function to restart services
function Restart-Services {
    param([string]$ServiceType)
    
    Write-Header "Restarting Services"
    docker-compose down
    Start-Services $ServiceType
}

# Function to show logs
function Show-Logs {
    param([string]$ServiceName)
    
    if ([string]::IsNullOrEmpty($ServiceName)) {
        Write-Header "Showing All Logs"
        docker-compose logs -f
    }
    else {
        Write-Header "Showing Logs for $ServiceName"
        docker-compose logs -f $ServiceName
    }
}

# Function to show status
function Show-Status {
    Write-Header "Service Status"
    docker-compose ps
}

# Function to clean up
function Clear-All {
    Write-Header "Cleaning Up"
    docker-compose down -v
    docker system prune -f
    Write-Status "Cleanup completed."
}

# Function to show help
function Show-Help {
    Write-Host "MASX Global Signal Grid Deployment Script" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\deploy.ps1 [COMMAND] [OPTIONS]" -ForegroundColor White
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor Yellow
    Write-Host "  build                    Build Docker image"
    Write-Host "  start [TYPE]             Start services"
    Write-Host "  stop                     Stop all services"
    Write-Host "  restart [TYPE]           Restart services"
    Write-Host "  logs [SERVICE]           Show logs"
    Write-Host "  status                   Show service status"
    Write-Host "  cleanup                  Clean up containers and volumes"
    Write-Host "  help                     Show this help message"
    Write-Host ""
    Write-Host "Service Types:" -ForegroundColor Yellow
    Write-Host "  fastapi                  FastAPI server only"
    Write-Host "  scheduler                Daily job scheduler only"
    Write-Host "  both|full                Both services (default)"
    Write-Host "  dev                      Development mode with hot reload"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Green
    Write-Host "  .\deploy.ps1 start both            Start both services"
    Write-Host "  .\deploy.ps1 start fastapi         Start FastAPI server only"
    Write-Host "  .\deploy.ps1 logs masx-full        Show logs for full service"
    Write-Host "  .\deploy.ps1 restart dev           Restart in development mode"
}

# Main script logic
switch ($Command) {
    "build" {
        Test-Prerequisites
        Build-Image
    }
    "start" {
        Test-Prerequisites
        Start-Services $ServiceType
    }
    "stop" {
        Stop-Services
    }
    "restart" {
        Test-Prerequisites
        Restart-Services $ServiceType
    }
    "logs" {
        Show-Logs $ServiceType
    }
    "status" {
        Show-Status
    }
    "cleanup" {
        Clear-All
    }
    "help" {
        Show-Help
    }
    default {
        Write-Error "Unknown command: $Command"
        Write-Host ""
        Show-Help
        exit 1
    }
} 