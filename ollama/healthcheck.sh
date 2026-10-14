#!/bin/sh
# Healthy only once the model named in params.yaml is pulled and usable, not
# merely once the server answers: the backend starts on "healthy" and its first
# request would otherwise hit a server that is still downloading the model.
# `ollama show` exits non-zero until the model's manifest is complete, and
# matches the exact name (a substring match on `ollama list` would accept
# qwen3:8b-q4 for qwen3:8b).
. /llm_model.sh
MODEL=$(read_llm_model)
[ -n "$MODEL" ] && ollama show "$MODEL" > /dev/null 2>&1
