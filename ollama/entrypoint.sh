#!/bin/sh
set -e

# Start Ollama server in the background
ollama serve &
SERVER_PID=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama server..."
until ollama list > /dev/null 2>&1; do
    sleep 1
done
echo "Ollama server is ready"

# Pull the model named in params.yaml -- the single source of truth.
. /llm_model.sh
MODEL=$(read_llm_model)

if [ -z "$MODEL" ]; then
    echo "ERROR: could not read llm.model from $PARAMS_FILE" >&2
    exit 1
fi
echo "Model from params.yaml: $MODEL"
if ! ollama list | grep -q "$MODEL"; then
    echo "Pulling model: $MODEL"
    ollama pull "$MODEL"
    echo "Model $MODEL is ready"
else
    echo "Model $MODEL already available"
fi

# Wait for the server process
wait $SERVER_PID
