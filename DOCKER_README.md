# Docker Deployment Guide for MASX Global Signal Grid

This guide explains how to deploy the MASX Global Signal Grid (MASX) Agentic AI System using Docker and Docker Compose.

## 🐳 Overview

The application has two main components:
1. **FastAPI Server** (`main_fast_api.py`) - REST API for data retrieval
2. **Daily Job Scheduler** (`daily_job.py`) - Scheduled execution of LangGraph workflows

## 📋 Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- Environment variables configured (see `.env` file)

## 🚀 Quick Start

### Option 1: Run Both Services (Recommended)
```bash
# Start both FastAPI server and scheduler
docker-compose up masx-full

# Or in detached mode
docker-compose up -d masx-full
```

### Option 2: Run Services Separately
```bash
# FastAPI server only
docker-compose up masx-fastapi

# Scheduler only
docker-compose up masx-scheduler
```

### Option 3: Development Mode
```bash
# Development with hot reload
docker-compose up masx-dev
```

## 🔧 Service Configurations

### 1. FastAPI Server Only (`masx-fastapi`)
- **Port**: 8000
- **Purpose**: REST API endpoints for data retrieval
- **Health Check**: Available at `http://localhost:8000/health`

### 2. Daily Job Scheduler Only (`masx-scheduler`)
- **Purpose**: Runs scheduled LangGraph workflows
- **Database**: Uses SQLite (`jobs.db`) for job persistence
- **Schedule**: Daily at midnight UTC (configurable)

### 3. Both Services (`masx-full`)
- **Port**: 8000
- **Purpose**: Complete system with API and scheduling
- **Process Management**: FastAPI runs in background, scheduler in foreground

### 4. Development Mode (`masx-dev`)
- **Port**: 8000
- **Hot Reload**: Enabled for code changes
- **Volume Mounts**: Source code mounted for live development

## 🌍 Environment Variables

Create a `.env` file with your configuration:

```bash
# Database Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=false

# LLM Configuration
OPENAI_API_KEY=your_openai_key
MISTRAL_API_KEY=your_mistral_key

# Logging
LOG_LEVEL=INFO

# Service Type (optional, defaults to "both")
SERVICE_TYPE=both
```

## 📊 Monitoring and Health Checks

### Health Check Endpoint
```bash
curl http://localhost:8000/health
```

### View Logs
```bash
# All services
docker-compose logs

# Specific service
docker-compose logs masx-full
docker-compose logs masx-fastapi
docker-compose logs masx-scheduler

# Follow logs
docker-compose logs -f masx-full
```

### Container Status
```bash
docker-compose ps
```

## 🔄 Service Management

### Start Services
```bash
# Start all services
docker-compose up

# Start specific service
docker-compose up masx-full

# Start in background
docker-compose up -d masx-full
```

### Stop Services
```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Restart Services
```bash
# Restart specific service
docker-compose restart masx-full

# Rebuild and restart
docker-compose up --build masx-full
```

## 🛠️ Customization

### Modify Scheduler Configuration
Edit `daily_job.py` to change scheduling:

```python
# Daily cron job: UTC midnight
scheduler.add_job(
    run_gsg_workflow,
    trigger='cron',
    hour=0,
    minute=0,
    id='masx_gsg_daily',
    replace_existing=True
)
```

### Change Service Type
Set environment variable:
```bash
export SERVICE_TYPE=fastapi  # or scheduler, both
```

### Custom Port Mapping
Edit `docker-compose.yml`:
```yaml
ports:
  - "8080:8000"  # Host port 8080, container port 8000
```

## 📁 Volume Mounts

The following directories are mounted:
- `./logs` → `/app/logs` - Application logs
- `./data` → `/app/data` - Data storage
- `./jobs.db` → `/app/jobs.db` - Scheduler database

## 🔍 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Check what's using port 8000
   lsof -i :8000
   
   # Use different port
   docker-compose up -p 8080 masx-fastapi
   ```

2. **Environment Variables Missing**
   ```bash
   # Check environment
   docker-compose exec masx-full env
   
   # Copy env.example to .env
   cp env.example .env
   ```

3. **Database Connection Issues**
   ```bash
   # Check logs
   docker-compose logs masx-full
   
   # Verify environment variables
   docker-compose exec masx-full printenv | grep SUPABASE
   ```

4. **Permission Issues**
   ```bash
   # Fix file permissions
   sudo chown -R $USER:$USER logs/ data/
   ```

### Debug Mode
```bash
# Run with debug output
docker-compose up --verbose masx-full

# Access container shell
docker-compose exec masx-full bash
```

## 🚀 Production Deployment

### Using Docker Run
```bash
# Build image
docker build -t masx-ai .

# Run both services
docker run -d \
  --name masx-full \
  -p 8000:8000 \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/jobs.db:/app/jobs.db \
  --env-file .env \
  masx-ai

# Run FastAPI only
docker run -d \
  --name masx-fastapi \
  -p 8000:8000 \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data:/app/data \
  --env-file .env \
  -e SERVICE_TYPE=fastapi \
  masx-ai
```

### Using Docker Swarm
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml masx
```

## 📈 Scaling

### Scale FastAPI Service
```bash
# Scale to 3 instances
docker-compose up --scale masx-fastapi=3
```

### Load Balancer Configuration
Add nginx or traefik for load balancing multiple FastAPI instances.

## 🔐 Security Considerations

1. **Non-root User**: Container runs as `masx` user
2. **Environment Variables**: Sensitive data in `.env` file
3. **Network Isolation**: Services use dedicated network
4. **Health Checks**: Automatic health monitoring

## 📞 Support

For issues or questions:
- Check logs: `docker-compose logs`
- Review configuration: `.env` file
- Verify prerequisites: Docker and Docker Compose versions
- Contact: ab@masxai.com 