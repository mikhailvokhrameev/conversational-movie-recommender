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

# Pull the model named in params.yaml -- the single source of truth. Parsed
# with awk rather than a YAML library to avoid adding one to this image: take
# the first "  model:" line inside the top-level "llm:" block.
PARAMS_FILE="${PARAMS_FILE:-/params.yaml}"
MODEL=$(awk '/^llm:/{inblock=1; next} /^[a-z]/{inblock=0} inblock && /^[[:space:]]+model:[[:space:]]*/{sub(/^[[:space:]]+model:[[:space:]]*/, ""); print; exit}' "$PARAMS_FILE")

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
