# Start from the Ollama base image (includes server + model support)
FROM ollama/ollama:latest

# Install Python + pip + system tools
RUN apt-get update && apt-get install -y \
    python3 python3-pip build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Set up app directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Make the Ollama + app launcher script executable
RUN chmod +x /app/init-ollama.sh

# Expose the Flask port
ENV PORT=10000
EXPOSE $PORT

# Start Ollama and Flask app using the entrypoint script
ENTRYPOINT ["/bin/sh", "/app/init-ollama.sh"]
