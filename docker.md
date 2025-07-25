
## **📁 Docker File Structure **

### **1. `Dockerfile` - The Foundation**

**Purpose**: Defines how to build the Docker image for your application.

**Key Components**:
```dockerfile
# Base image - Python 3.12 slim for smaller size
FROM python:3.12-slim

# Environment variables for Python optimization
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    sqlite3 \  # Added for scheduler database

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model for NLP
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY src/ ./src/
COPY *.py ./  # Copy main entry points
COPY pyproject.toml .
COPY env.example .

# Create directories and set permissions
RUN mkdir -p logs data
RUN useradd --create-home --shell /bin/bash masx && \
    chown -R masx:masx /app
USER masx

# Set Python path
ENV PYTHONPATH=/app/src

# Copy and set up entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Health check for FastAPI
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use entrypoint script
ENTRYPOINT ["docker-entrypoint.sh"]
```

**Why This Design**:
- **Multi-stage ready**: Can be extended for production optimization
- **Security**: Runs as non-root user
- **Efficiency**: Optimized layer caching
- **Flexibility**: Entrypoint allows different service modes

---

### **2. `docker-entrypoint.sh` - The Smart Launcher**

**Purpose**: Determines which service(s) to start based on environment variables.

**How It Works**:
```bash
#!/bin/bash

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

# Function to start both services
start_both() {
    echo "�� Starting both FastAPI server and daily job scheduler..."
    cd /app
    
    # Start FastAPI in background
    python main_fast_api.py &
    FASTAPI_PID=$!
    
    # Wait for FastAPI to start
    sleep 5
    
    # Start scheduler in foreground
    python daily_job.py &
    SCHEDULER_PID=$!
    
    # Handle graceful shutdown
    shutdown() {
        echo "🛑 Shutting down services..."
        kill $FASTAPI_PID 2>/dev/null || true
        kill $SCHEDULER_PID 2>/dev/null || true
        exit 0
    }
    
    trap shutdown SIGTERM SIGINT
    wait $FASTAPI_PID $SCHEDULER_PID
}

# Main logic based on SERVICE_TYPE environment variable
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
        exit 1
        ;;
esac
```

**Key Features**:
- **Service Selection**: Uses `SERVICE_TYPE` environment variable
- **Process Management**: Proper PID tracking and signal handling
- **Graceful Shutdown**: Handles SIGTERM/SIGINT properly
- **Background/Foreground**: FastAPI runs in background, scheduler in foreground

---

### **3. `docker-compose.yml` - Service Orchestration**

**Purpose**: Defines different service configurations for various deployment scenarios.

**Service Configurations**:

#### **FastAPI Server Only (`masx-fastapi`)**
```yaml
masx-fastapi:
  build: .
  ports:
    - "8000:8000"
  environment:
    - SERVICE_TYPE=fastapi
  volumes:
    - ./logs:/app/logs
    - ./data:/app/data
  env_file:
    - .env
  restart: unless-stopped
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
    interval: 30s
    timeout: 10s
    retries: 3
```

#### **Daily Job Scheduler Only (`masx-scheduler`)**
```yaml
masx-scheduler:
  build: .
  environment:
    - SERVICE_TYPE=scheduler
  volumes:
    - ./logs:/app/logs
    - ./data:/app/data
    - ./jobs.db:/app/jobs.db  # Persistent scheduler database
  env_file:
    - .env
  restart: unless-stopped
```

#### **Both Services (`masx-full`)**
```yaml
masx-full:
  build: .
  ports:
    - "8000:8000"
  environment:
    - SERVICE_TYPE=both
  volumes:
    - ./logs:/app/logs
    - ./data:/app/data
    - ./jobs.db:/app/jobs.db
  env_file:
    - .env
  restart: unless-stopped
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
```

#### **Development Mode (`masx-dev`)**
```yaml
masx-dev:
  build: .
  ports:
    - "8000:8000"
  environment:
    - SERVICE_TYPE=fastapi
    - API_RELOAD=true
  volumes:
    - ./src:/app/src  # Live code mounting
    - ./logs:/app/logs
    - ./data:/app/data
    - ./jobs.db:/app/jobs.db
  env_file:
    - .env
  restart: unless-stopped
```

**Why Multiple Services**:
- **Separation of Concerns**: Run services independently
- **Resource Optimization**: Only run what you need
- **Development Flexibility**: Different modes for different use cases
- **Scaling**: Can scale services independently

---

### **4. `.dockerignore` - Build Optimization**

**Purpose**: Excludes unnecessary files from Docker build context, making builds faster and more secure.

**Key Exclusions**:
```dockerignore
# Git files
.git
.gitignore

# Python cache and virtual environments
__pycache__
*.pyc
.env
.venv
venv/

# IDE files
.vscode/
.idea/

# OS files
.DS_Store
Thumbs.db

# Docker files themselves
Dockerfile*
docker-compose*
.dockerignore

# Documentation and tests
README*
*.md
tests/
test_*

# Logs and data (mounted as volumes)
logs/
*.log
data/
*.db
jobs.db
```

**Benefits**:
- **Faster Builds**: Smaller build context
- **Security**: Excludes sensitive files
- **Efficiency**: Prevents unnecessary file copying

---

### **5. `deploy.sh` - Linux/macOS Deployment Script**

**Purpose**: Provides easy-to-use commands for deployment management.

**Key Functions**:

#### **Prerequisites Check**
```bash
check_prerequisites() {
    # Check Docker installation
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed."
        exit 1
    fi
    
    # Check .env file
    if [ ! -f ".env" ]; then
        if [ -f "env.example" ]; then
            cp env.example .env
            print_status "Please edit .env file with your configuration."
        fi
    fi
}
```

#### **Service Management**
```bash
start_services() {
    local service_type=$1
    
    case $service_type in
        "fastapi")
            docker-compose up -d masx-fastapi
            ;;
        "scheduler")
            docker-compose up -d masx-scheduler
            ;;
        "both"|"full")
            docker-compose up -d masx-full
            ;;
        "dev")
            docker-compose up -d masx-dev
            ;;
    esac
}
```

**Usage Examples**:
```bash
./deploy.sh start both      # Start both services
./deploy.sh start fastapi   # Start FastAPI only
./deploy.sh logs masx-full  # View logs
./deploy.sh status          # Check status
./deploy.sh stop            # Stop all services
```

---

### **6. `deploy.ps1` - Windows PowerShell Script**

**Purpose**: Windows equivalent of the deployment script.

**Key Differences from Bash**:
```powershell
# PowerShell parameter handling
param(
    [Parameter(Position=0)]
    [string]$Command = "help",
    
    [Parameter(Position=1)]
    [string]$ServiceType = "both"
)

# PowerShell-specific commands
function Test-Prerequisites {
    try {
        docker --version | Out-Null
    }
    catch {
        Write-Error "Docker is not installed."
        exit 1
    }
}

# PowerShell switch statement
switch ($Command) {
    "build" {
        Test-Prerequisites
        Build-Image
    }
    "start" {
        Test-Prerequisites
        Start-Services $ServiceType
    }
}
```

**Usage Examples**:
```powershell
.\deploy.ps1 start both      # Start both services
.\deploy.ps1 start fastapi   # Start FastAPI only
.\deploy.ps1 logs masx-full  # View logs
.\deploy.ps1 status          # Check status
```

---

### **7. `DOCKER_README.md` - Comprehensive Documentation**

**Purpose**: Complete deployment guide with examples, troubleshooting, and best practices.

**Sections Include**:
- **Overview**: System architecture explanation
- **Prerequisites**: Required software and setup
- **Quick Start**: Immediate deployment commands
- **Service Configurations**: Detailed explanation of each service
- **Environment Variables**: Configuration options
- **Monitoring**: Health checks and logging
- **Troubleshooting**: Common issues and solutions
- **Production Deployment**: Advanced deployment scenarios
- **Security Considerations**: Best practices

---

## **�� How All Files Work Together**

### **Deployment Flow**:

1. **User runs deployment command**:
   ```bash
   ./deploy.sh start both
   ```

2. **Deploy script checks prerequisites**:
   - Docker installation
   - Docker Compose installation
   - Environment file existence

3. **Docker Compose reads configuration**:
   - Uses `docker-compose.yml` to determine service configuration
   - Sets `SERVICE_TYPE=both` environment variable

4. **Docker builds image**:
   - Uses `Dockerfile` to build container image
   - `.dockerignore` excludes unnecessary files
   - Installs dependencies and copies code

5. **Container starts**:
   - `docker-entrypoint.sh` is executed
   - Reads `SERVICE_TYPE` environment variable
   - Starts appropriate service(s)

6. **Services run**:
   - FastAPI server starts on port 8000
   - Scheduler starts with job database
   - Health checks monitor service status

### **File Dependencies**:

```
User Command
    ↓
deploy.sh/deploy.ps1
    ↓
docker-compose.yml
    ↓
Dockerfile
    ↓
docker-entrypoint.sh
    ↓
Application Services
```

### **Environment Variables Flow**:

```
.env file
    ↓
docker-compose.yml (env_file)
    ↓
Container environment
    ↓
docker-entrypoint.sh (SERVICE_TYPE)
    ↓
Service selection
```

## **🎯 Why This Many Files?**

### **1. Separation of Concerns**
- **Dockerfile**: Image building
- **docker-compose.yml**: Service orchestration
- **entrypoint.sh**: Service selection logic
- **deploy scripts**: User interface

### **2. Cross-Platform Support**
- **deploy.sh**: Linux/macOS users
- **deploy.ps1**: Windows users
- **docker-compose.yml**: Universal Docker Compose

### **3. Flexibility**
- **Multiple service configurations**: Different deployment scenarios
- **Environment-based selection**: Runtime service choice
- **Development vs Production**: Different modes

### **4. Maintainability**
- **Modular design**: Each file has a single responsibility
- **Clear documentation**: Comprehensive README
- **Easy troubleshooting**: Isolated components

### **5. Production Readiness**
- **Health checks**: Service monitoring
- **Graceful shutdown**: Proper signal handling
- **Security**: Non-root user, environment isolation
- **Logging**: Persistent log storage

This multi-file approach provides a robust, flexible, and maintainable containerization solution that can handle various deployment scenarios while remaining easy to use and understand.