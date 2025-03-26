#!/bin/sh

set -e  # Stop script on first error

export OLLAMA_NOPRUNE=false

# Start Ollama in background
ollama serve &
OLLAMA_PID=$!

echo "🦙 Waiting for Ollama to start..."
sleep 10

echo "📦 Pulling CodeLlama 7B model..."
ollama pull codellama:7b

echo "🚀 Starting Flask app with Gunicorn..."
# Start gunicorn in foreground so Render can track it
exec gunicorn --bind 0.0.0.0:$PORT --timeout 600 --log-level debug app:app
