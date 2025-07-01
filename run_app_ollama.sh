#!/bin/bash

# Activate virtual environment (cross-platform compatible)
if [ -f ./.venv/bin/activate ]; then
    # Unix/Linux/WSL
    source ./.venv/bin/activate
elif [ -f ./.venv/Scripts/activate ]; then
    # Windows
    source ./.venv/Scripts/activate
else
    echo "Virtual environment not found. Please create one first."
    exit 1
fi

# Create .models directory if it doesn't exist
mkdir -p .models

# export OLLAMA_FLASH_ATTENTION=1       # uncomment if you wanna use flash attention
# export OLLAMA_KV_CACHE_TYPE=q8_0      # uncomment if you wanna use a quantized version of the model


# Check if Ollama is already running, if not start it
if command -v pgrep &> /dev/null; then
    if ! pgrep -x "ollama" > /dev/null; then
        echo "Starting Ollama server..."
        ollama serve &
        sleep 3
    else
        echo "Ollama server is already running"
    fi
fi

ollama pull "gemma3n:latest" &&
streamlit run ./chatbot_app.py local
