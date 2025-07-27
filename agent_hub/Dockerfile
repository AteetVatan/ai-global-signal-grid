# Use minimal Python image
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential libffi-dev libxml2-dev libxslt1-dev \
    libjpeg-dev zlib1g-dev git curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Run the daily job script
CMD ["python", "daily_job.py"]
