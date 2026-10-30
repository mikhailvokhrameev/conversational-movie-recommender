# Infrastructure and Deployment

## Modes and configuration

Both modes use Django/uvicorn, PostgreSQL 16/pgvector and React/Vite with the same public HTTP/SSE contract. Prompts, intent validation, retrieval, scoring, normalization, blending, relevance gate, MMR, explanations, history and stream splitting are shared. The cached adapter factory selects providers once per process. Restart after changing configuration; there is no automatic provider fallback.

| Component | local (default) | openrouter |
|---|---|---|
| LLM | Ollama `qwen3.5:9b-q4_K_M` | `qwen/qwen3.5-9b` |
| Embedding | pinned `BAAI/bge-m3` | `baai/bge-m3`, DeepInfra routing with fallback disabled |
| Reranking | pinned `BAAI/bge-reranker-v2-m3` | `voyageai/rerank-3-lite` |

`params.yaml` is the single source of truth for models, revisions, dimensions, timeouts, retries, limits and scoring weights. `runtime_profiles.openrouter` overlays the shared settings. The canonical embedding identity remains `BAAI/bge-m3`, dimensions 1024; API model ID is separate and unknown API revision is null. Cloud reranking starts at top_k 50, weight 0.5, min_score null: these are uncalibrated starting values.

Environment variables are secrets/wiring only: `DB_PASSWORD`, `SECRET_KEY`, `DEBUG`, `INFERENCE_MODE` (local/openrouter), `OPENROUTER_API_KEY`, `PARAMS_PATH`. See `.env.example`; users manage their own `.env`. Compose selects the mode in its launch files, so changing only `.env` does not switch a Compose deployment. `PARAMS_PATH` defaults to `/app/params.yaml` in Compose and must point to an accessible file (mount a custom file explicitly if needed). Host management commands use `DATABASE_URL` for database wiring; Compose builds that URL for its own DB. `ALLOWED_HOSTS` and `CORS_ALLOWED_ORIGINS` are literal Compose settings.

## User launch and switching commands

**Only the user runs the commands in this document.** They can download models, load GPU weights, contact real providers and incur API costs. Agents must never execute real API calls, preflight, application startup, downloads or Docker integration checks; offline tests mock HTTP and block networking.

Shared `docker-compose.yml` defines db, backend and frontend. Automatic `docker-compose.override.yml` adds Ollama and NVIDIA reservations to Ollama/backend. Explicit cloud `-f` selection excludes the automatic override, starts no Ollama, requests no GPU and loads no local ML models. Cloud retains the common backend image, including local dependencies.

```bash
# Local: NVIDIA driver/container toolkit required; first start pulls models.
docker compose up --build
# In another terminal:
docker compose exec backend python manage.py check_inference
docker compose exec backend python manage.py generate_embeddings

# Before local -> cloud: preserve all volumes (NO -v).
docker compose down
# Set OPENROUTER_API_KEY yourself in .env, then:
docker compose -f docker-compose.yml -f docker-compose.openrouter.yml up --build
docker compose -f docker-compose.yml -f docker-compose.openrouter.yml exec backend python manage.py check_inference

# Before cloud -> local: use the CURRENT mode's files, again NO -v.
docker compose -f docker-compose.yml -f docker-compose.openrouter.yml down
docker compose up --build
```

Backend startup is `migrate -> import_catalog --skip-existing -> check_embedding_index -> uvicorn`. Embedding generation is always explicit, never automatic. An empty index warns; startup never silently rebuilds it. Backend waits for healthy DB in both modes and healthy Ollama only locally. Ollama health requires the exact `llm.model` to be present; its startup grace period is 900s.

Host ports: frontend 3000, backend 8000, PostgreSQL **5433** (internal 5432), local-only Ollama 11434. `pgdata`, `ollama_data` and `model_cache` persist across `down`; eval records persist under `./eval_runs`. Published DB/Ollama ports assume a trusted local machine; restrict exposure before deploying elsewhere.

## Embedding lifecycle and recovery

`EmbeddingIndex` records canonical model, dimensions, text template version/settings, revision, provider and API provider. Migration 0010 preserves old local provenance. Model/dimensions changes require a schema migration and explicit full regeneration; text changes require a template-version bump and full regeneration. Different non-null revisions fail; a null on either side only warns, never inventing API weight provenance.

After an origin change, `check_embedding_index` re-embeds a deterministic sample in tmdb_id order and checks the configured compatibility threshold. This is read-only for the database but may make real embedding API calls at startup. A pass permits reading that index only; it neither proves every vector identical nor changes its origin. Partial generation still refuses to mix origins, even after a passing check.

If compatibility fails or is unavailable, startup stops. Restore the original mode/configuration to recover service. Alternatively, after an explicit user decision and backup, run `generate_embeddings --all` against the intended catalog to replace every vector and provenance; never run it automatically or merely to test cloud. `--adopt` applies only to legacy vectors without provenance whose origin the user knows; it cannot relabel an existing tracked index.

### Separate cloud test database (user only)

Use an isolated Compose project with its own volumes; stop the main project first to avoid host-port conflicts. This exercises the complete cloud lifecycle without rebuilding the working local catalog:

```bash
docker compose down
docker compose -p movie-cloud-test -f docker-compose.yml -f docker-compose.openrouter.yml up --build
docker compose -p movie-cloud-test -f docker-compose.yml -f docker-compose.openrouter.yml exec backend python manage.py check_inference
docker compose -p movie-cloud-test -f docker-compose.yml -f docker-compose.openrouter.yml exec backend python manage.py generate_embeddings
docker compose -p movie-cloud-test -f docker-compose.yml -f docker-compose.openrouter.yml exec backend python manage.py check_embedding_index
# Also check a second generation run only fills missing rows and preserves origin.
docker compose -p movie-cloud-test -f docker-compose.yml -f docker-compose.openrouter.yml exec backend python manage.py generate_embeddings
docker compose -p movie-cloud-test -f docker-compose.yml -f docker-compose.openrouter.yml down
```

Separately start cloud on the original project to read-check its existing local index using the startup compatibility check; do not run generation there. Record pass/failure and confirm vector count/provenance unchanged. Restore local mode after stopping cloud without `-v`. Keep test and working catalogs separate.

## Reproducibility and pending manual acceptance

**Real validation is pending. Offline tests do not establish API availability, provider semantics, GPU compatibility, memory fit, speed or recommendation quality.**

Local weights keep their pinned HF revisions, `torch==2.5.1`, CUDA 12.1 and transformers constraints. Python dependencies remain locked in `backend/uv.lock`. The Ollama server image is pinned in `ollama/Dockerfile` to `ollama/ollama:0.34.2`; compatibility of that pin with the new Qwen3.5 tag and required API behavior is not yet confirmed. Tags are mutable; record the installed server version and served model digest. API routing/model revisions can differ from local weights even under the same canonical name.

The user must record:

- GTX 1080 Ti: all three models resident together at context **6144**, peak VRAM (`nvidia-smi`), GPU offload, Ollama version/model digest, cold/warm TTFT and end-to-end latency. Test a long conversation, top-10 results, schema-valid JSON, think suppression and complete explanations. Historical Qwen3 8B memory/timing estimates do not validate Qwen3.5 9B. Insufficient VRAM is a blocking result; do not automatically reduce the agreed model or context.
- Without GPU: real cloud preflight, full initial embedding lifecycle on the separate test DB, SSE search/refinement/follow-up, public API/history behavior, and embedding failure returning 503. Confirm no local model loads/Ollama/GPU reservations. Separately test compatibility reading the existing local index.
- Quality: dev evaluations in both modes on identical data, separate run records, mode/provider/config-hash metadata and failed-run diagnostics. Compare intent errors, parse fallbacks, empty results and latency; inspect recommendation/explanation quality manually. Preserve existing evaluation flags and recommendation-quality tasks 5–8. Voyage gate stays null until calibrated from user relevance labels; LLM judge is circular and never ground truth.

Use the selected mode's Compose prefix for these user commands:

```bash
docker compose exec backend python manage.py evaluate_scoring --split dev
docker compose exec backend python manage.py evaluate_scoring --split dev --intent oracle
docker compose exec backend python manage.py evaluate_scoring --intent oracle --k 10 --label-export eval_runs/labels.csv
# Offline labelled curves, after user hand-labelling:
cd backend && python manage.py evaluate_scoring --label-report ../eval_runs/labels.csv
```

Do not interpret failed provider calls as a successful quality run. Keep mode-specific label-export filenames to avoid overwriting labels; run JSON records carry mode, providers and effective-config hash.

## Management Commands

Run inside the backend container: `docker compose exec backend python manage.py <command>`

| Command | Purpose |
|---------|---------|
| `import_catalog --skip-existing` | Load the TMDB CSV into PostgreSQL (idempotent), then rebuild search vectors |
| `generate_embeddings` | Embed movies that have no embedding (batch size from params.yaml) and record the index provenance; refuses under a config mismatch |
| `generate_embeddings --all` | Re-embed every movie and refresh the provenance record (required after changing the embedding model or `embedding.text.*`) |
| `generate_embeddings --adopt` | Record the current config for an index built before provenance tracking, without re-embedding |
| `check_embedding_index` | Check provenance; origin changes probe a deterministic sample through the active embedder (may incur API costs), without writing vectors |
| `check_inference` | User-only real probes through the active LLM, embedder and reranker; not automatic startup |
| `evaluate_scoring` | Golden-set evaluation on the test split, intent parsed by the production LLM call; writes a run record to `eval_runs/` |
| `evaluate_scoring --intent oracle` | Same, with each entry's labelled intent instead of the LLM parse |
| `evaluate_scoring --llm-judge` | Also score results with the LLM judge (slow, secondary signal) |
| `evaluate_scoring --sweep` | Grid search over scoring weight space, dev split only |
| `evaluate_scoring --sweep --llm-judge` | Grid search with LLM judge columns (6 configs) |
| `evaluate_scoring --intent oracle --gate-report` | Calibration report for `reranking.min_score` (gate off; dev split only) |
| `evaluate_scoring --intent oracle --k 10 --label-export eval_runs/labels.csv` | CSV of ungated dev results for hand-labelling relevance (dev split only; `eval_runs/` is the container's writable mount, `./eval_runs` on the host) |
| `evaluate_scoring --label-report ../eval_runs/labels.csv` | Threshold/margin curves from the labelled CSV, run on the host from `backend/` (no DB, Ollama or models) |
| `cleanup_sessions` | Delete expired chat sessions (>24h) |
