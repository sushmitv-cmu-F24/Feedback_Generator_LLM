#!/bin/sh

# Start Ollama in background
ollama serve &
OLLAMA_PID=$!

echo "🦙 Waiting for Ollama to start..."
sleep 15

echo "📦 Pulling CodeLlama 7B model..."
ollama pull codellama:7b

echo "🚀 Starting Flask app with Gunicorn..."
exec gunicorn --bind 0.0.0.0:$PORT --timeout 600 app:app

wait $OLLAMA_PID
