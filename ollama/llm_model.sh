#!/bin/sh
# Sourced by entrypoint.sh and healthcheck.sh so both read the model name the
# same way.

# Print llm.model from params.yaml -- the single source of truth. Parsed with
# awk rather than a YAML library to avoid adding one to this image: take the
# first "  model:" line inside the top-level "llm:" block, minus quotes and a
# trailing comment.
read_llm_model() {
    PARAMS_FILE="${PARAMS_FILE:-/params.yaml}"
    awk '/^llm:/{inblock=1; next} /^[a-z]/{inblock=0} inblock && /^[[:space:]]+model:[[:space:]]*/{sub(/^[[:space:]]+model:[[:space:]]*/, ""); sub(/[[:space:]]+#.*$/, ""); gsub(/["'"'"']/, ""); sub(/[[:space:]]+$/, ""); print; exit}' "$PARAMS_FILE"
}
