# Task 8 report

Status: DONE_WITH_CONCERNS (full-suite PostgreSQL setup unavailable).

Implemented shared db/backend/frontend base, automatic local override with Ollama and both NVIDIA reservations, and explicit OpenRouter overlay with backend-only API key. Preserved implicit directory-based project naming, pgdata/ollama_data/model_cache keys, shared backend build, and migrate/import/check-index startup. No embedding rebuild or preflight added. Updated only .env.example for environment documentation; no user .env read or changed. Ollama pin unchanged.

## TDD and static validation evidence

From backend, RED command:
`UV_OFFLINE=1 UV_CACHE_DIR=/private/tmp/movie-uv-cache PYTHONPATH=/private/tmp uv run --no-sync pytest -p movie_offline_guard core/tests/test_compose_config.py -q`

Before implementation: `2 failed in 1.86s`: local failed with `KeyError: 'PARAMS_PATH'`; cloud failed because docker-compose.openrouter.yml did not exist. Expected missing configuration failures.

GREEN same focused command: `2 passed in 0.70s`.

Tests execute only `docker compose --env-file <temporary fixture> [explicit -f files for cloud] config --format json`, never engine operations. Child environment contains only COMPOSE_DISABLE_ENV_FILE=1 and a fixed executable PATH, without inheriting shell secrets. The temporary fixture contains fictitious DB/Django/OpenRouter secrets; resolved config is captured, not printed. Default-file discovery verifies automatic local override; explicit base/cloud verifies exclusion of local override. Assertions cover local Ollama and two GPU reservations; cloud absence of Ollama/GPU/dependency; key scoped to backend; common startup; backend build; implicit volume names. Compose omits unused ollama_data from cloud resolved output, while its original declaration remains in base.

Full-suite command:
`UV_OFFLINE=1 UV_CACHE_DIR=/private/tmp/movie-uv-cache PYTHONPATH=/private/tmp uv run --no-sync pytest -p movie_offline_guard -q`
Result: `1272 passed, 1 warning, 78 errors in 12.57s`. All errors are PostgreSQL database setup failures (connection unavailable at localhost); no Docker engine/container was started. The full suite is not claimed green.

`git diff --check`: passed.

## Changed files

- docker-compose.yml
- docker-compose.override.yml
- docker-compose.openrouter.yml
- .env.example
- backend/core/tests/test_compose_config.py
- this report

## Self-review

Reviewed diff and new files. Removed stale GPU-only comments from shared base. Original volume declarations and implicit project naming retained. Both modes use exactly the existing backend build and startup chain. No models, parameters, pin, application code, docs, real API calls, downloads or engine operations changed/performed. Config tests skip explicitly if Docker CLI absent, and otherwise require Compose's static config command.

## Concerns

Real GPU/Ollama Qwen3.5 compatibility, full database tests, application launches, and provider calls remain user-run checks under the task constraints. Static config checks passed; full-suite DB setup unavailable as above.
