#!/bin/bash
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to start
echo "Waiting for Ollama to start..."
sleep 15

# Always pull codellama:7b
echo "Pulling CodeLlama 7B model..."
ollama pull codellama:7b

# Keep the container running
wait $OLLAMA_PID
