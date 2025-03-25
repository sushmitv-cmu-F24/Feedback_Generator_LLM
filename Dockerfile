FROM ollama/ollama:latest as ollama

FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy Ollama from the ollama image
COPY --from=ollama /bin/ollama /bin/ollama

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy application code and data files
COPY . .

# Create a startup script
RUN echo '#!/bin/bash\n\
# Start Ollama in the background\n\
ollama serve > /dev/null 2>&1 &\n\
OLLAMA_PID=$!\n\
\n\
# Wait for Ollama to start\n\
echo "Starting Ollama server..."\n\
sleep 5\n\
\n\
# Pull the model if not already present (runs only once)\n\
echo "Checking for CodeLlama model..."\n\
if ! ollama list | grep -q "codellama"; then\n\
    echo "Pulling CodeLlama model (this may take several minutes)..."\n\
    ollama pull codellama:13b\n\
fi\n\
\n\
# Make sure data directories have correct permissions\n\
chmod -R 755 /app/data\n\
\n\
# Start the Flask application\n\
echo "Starting Flask application..."\n\
exec gunicorn --bind 0.0.0.0:$PORT --timeout 300 app:app\n\
' > start.sh && chmod +x start.sh

# Set environment variable for port
ENV PORT=10000

# Expose the port
EXPOSE $PORT

# Start the application
CMD ["./start.sh"]