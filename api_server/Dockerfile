# Use a minimal Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port required
EXPOSE 8000

# Run FastAPI app via uvicorn
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
CMD ["python", "main.py"]


