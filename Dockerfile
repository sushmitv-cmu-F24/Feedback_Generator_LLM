# No need to copy or start Ollama in this Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn psutil

# Copy application code
COPY . .

# Set environment variable for port
ENV PORT=10000

# Expose the port
EXPOSE $PORT

# Start the application
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--timeout", "600", "app:app"]