# Infrastructure and Deployment

## Modes and configuration

Both modes use Django/uvicorn, PostgreSQL 16/pgvector and React/Vite with the same public HTTP/SSE contract. Prompts, intent validation, retrieval, scoring, normalization, blending, relevance gate, MMR, explanations, history and stream splitting are shared. The cached adapter factory selects providers once per process. Restart after changing configuration; there is no automatic provider fallback.

| Component | local (default) | openrouter |
|---|---|---|
| LLM | Ollama `qwen3.5:9b-q4_K_M` | `qwen/qwen3.5-9b` |
| Embedding | pinned `BAAI/bge-m3` | `baai/bge-m3`, DeepInfra routing with fallback disabled |
| Reranking | pinned `BAAI/bge-reranker-v2-m3` | `voyageai/rerank-3-lite` |

`params.yaml` is the single source of truth for models, revisions, dimensions, timeouts, retries, limits and scoring weights. `runtime_profiles.openrouter` overlays the shared settings. The canonical embedding identity remains `BAAI/bge-m3`, dimensions 1024; API model ID is separate and unknown API revision is null. Cloud reranking starts at top_k 50, weight 0.5, min_score null: these are uncalibrated starting values.

Environment variables are secrets/wiring only: `DB_PASSWORD`, `SECRET_KEY`, `DEBUG`, `INFERENCE_MODE` (local/openrouter), `OPENROUTER_API_KEY`, `TMDB_API_KEY`, `PARAMS_PATH`. See `.env.example`; users manage their own `.env`. Compose selects the mode in its launch files, so changing only `.env` does not switch a Compose deployment. `PARAMS_PATH` defaults to `/app/params.yaml` in Compose and must point to an accessible file (mount a custom file explicitly if needed). Host management commands use `DATABASE_URL` for database wiring; Compose builds that URL for its own DB. `ALLOWED_HOSTS` and `CORS_ALLOWED_ORIGINS` are literal Compose settings.

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

Backend startup is `migrate -> import_catalog --skip-existing -> check_embedding_index -> sync_tmdb_catalog -> uvicorn`. Migrations, CSV import and embedding-index checks remain blocking. The TMDB update is best effort: missing key, API/preparation failure, or even an unexpected command failure allows serving the existing catalog. Startup never rebuilds the old index; an empty index needs explicit initial embedding preparation. Successful TMDB additions include newly generated embeddings. Backend waits for healthy DB in both modes and healthy Ollama only locally. Ollama health requires the exact `llm.model` to be present; its startup grace period is 900s.

Host ports: frontend 3000, backend 8000, PostgreSQL **5433** (internal 5432), local-only Ollama 11434. `pgdata`, `ollama_data` and `model_cache` persist across `down`; eval records persist under `./eval_runs`. Published DB/Ollama ports assume a trusted local machine; restrict exposure before deploying elsewhere.

## TMDB catalog update (user only)

Place the catalog CSV at `data/TMDB Movie Dataset v11.csv` and the optional evaluation
source at `data/Reddit Movie Large Test.csv`; both appear under `/app/data/` read-only.
The reviewed evaluation queries are `data/label_queries.json`, not a catalog input.
Obtain a TMDB v3 API key through your TMDB account API settings and set
`TMDB_API_KEY` yourself in `.env` using `.env.example`. Check TMDB's current terms,
licensing and attribution obligations yourself before using or distributing data.
Agents do not retrieve keys, contact TMDB or run these operations.

Both inference modes run the same sync on every backend startup. It traverses Discover
from 2024-01-01 through today's UTC date, then fetches English details/keywords. Every
run repeats that interval, so films previously below 20 votes can qualify later.
Only new IDs meeting the released/non-adult, description, poster and vote-count
filters are added; existing rows are unchanged. Discover may miss films (including
undated/unindexed films or changes during traversal); this is no completeness guarantee.
The API does not supply an atomic catalog snapshot.

Traversal and embeddings finish before atomic publication. Failed pagination,
invalid vectors or failed publication leave no partial new catalog. Missing key or
update failure produces a safe warning and retries on the next startup; the user can
also run `docker compose exec backend python manage.py sync_tmdb_catalog` with the
current mode's Compose prefix. Fix configuration/provider or index readiness first.
Sync adds startup delay and can incur cloud embedding costs; limits/timeouts/retries
live in `params.yaml`, while credentials remain environment wiring.

For an imported old catalog without ready vectors/provenance, first follow the
explicit embedding workflow below in its original inference mode. Sync skips when
that preparation is missing. A successful read-only origin compatibility check
still does not authorize partial generation from a different origin: sync skips
rather than mixing vectors. Restore the original mode, or use a separately approved
catalog rebuild; no automatic full regeneration occurs.

Before enabling updates on the working catalog, the user must use a separate test
DB/project to check the first 2024 backfill, repeated run, vote threshold crossing,
local/cloud behavior, TMDB outage and recovery. Record time, cost, observed coverage
and terms compliance. Separately read-check the existing index without rebuilding it.
Live API, PostgreSQL and GPU acceptance remain pending until user evidence exists.

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
| `sync_tmdb_catalog` | Best-effort startup/manual addition of eligible new TMDB films since 2024, with embeddings; failures preserve the existing catalog |
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

### Quality calibration operations

Only the user runs `evaluate_scoring --pool-export`: local capture requires models/GPU;
OpenRouter capture can incur paid calls. Offline `--grade-report`, `--fit` and
`--frozen-proposal` replay saved evidence without providers or DB. Preserve each CSV,
its adjacent `.run` directory and the source dev manifest at its recorded path. Use
separate mode-specific dev/test exports and freeze the dev proposal before test grades
are read; see the [workflow](ML.md#graded-saved-pool-workflow-quality-calibration-pending).
No graded quality acceptance or new calibrated settings are established yet.

### User-only review checks still pending

Run these checks separately for local and OpenRouter using the mode's Compose prefix above; keep all output in new mode-specific files under `eval_runs/`. None was run by an agent.

1. Database behavior: `docker compose exec backend pytest core/tests/test_catalog_bounds.py core/tests/test_candidate_generation.py -v`. `TestQualityChannel` uses mocks and does not establish database behavior. Run the real PostgreSQL edge scenarios specifically with `docker compose exec backend pytest core/tests/test_candidate_generation.py::TestQualityChannelPostgres -v`: one retains a low-vote semantic candidate after excluding the only high-vote movie, and one excludes the whole pool. Retain output for both cases and catalog-mean cache tests; results remain pending until supplied. The full DB acceptance command is `docker compose exec backend pytest -v`; see the host test record below for a separate PostgreSQL-backed run.
2. Prompt budget: capture `Inference usage component=llm` diagnostics for the same empty-history request and a conversation at the configured history limit; record provider input tokens, schema/repair overhead and remaining space within context 6144. For local diagnostics also retain Ollama `prompt_eval_count`. Compare the actual serialized classify payload with the `wants_obscure` schema property and example keys omitted in a separate measurement only; use the served model's tokenizer or provider usage, never chars/4. Do not change production prompts or context for this check.
3. Obscure-intent recall: run `aclassify_and_parse` and synchronous `parse_intent` on “obscure low-budget science fiction”, “hidden gems, not mainstream hits”, “lesser-known Korean thrillers”, plus controls “popular science fiction”, “highly rated Korean thrillers”, and “movies like Hidden Figures”. Repeat with history introducing and then withdrawing the obscure preference. Save raw response, extracted flag, fallback/error diagnostics and human expected flags; report false positives/negatives separately. Mocked propagation tests do not establish this recall.
4. Batch equivalence: through the cached adapters from `core.inference_backends.get_inference_backends().embedder` and `.reranker`, compare `encode_texts` on a fixed ordered list versus concatenated singleton results; compare `predict_scores` on the same query/documents together versus singleton documents. Include a duplicate, short text, empty text and long text at configured truncation. Record exact inputs, model/provider/revision, batch sizes, shapes, finite values, cosine/max absolute differences, rank changes, errors and timing. Repeat per mode without writing catalog vectors. Review observed tolerances rather than assuming numerical identity across providers.
5. Latency/GPU: run `check_inference`, then a cold and warm search, refinement and a long dialogue with top-10 explanations. Record Ollama version/digest, peak `nvidia-smi` VRAM, offload, TTFT and complete SSE latency with context 6144 and all models resident. Record JSON validation, think suppression and explanation completeness. Insufficient VRAM remains blocking. Cloud lifecycle must use the separate test project described above; existing-index checks are read-only.
6. Quality acceptance: use identical user-reviewed queries in separate mode-specific dev pools and human grade them before fitting/freezing; apply the frozen choice once to the independent test pool. Preserve all manifests, snapshots, worksheets and grades. Report nDCG, junk/empty rates, judged coverage and paired bootstrap intervals together; calibrate BGE and Voyage gates independently. A null Voyage gate and an LLM judge are not acceptance evidence. Task 8 cleanup stays pending until both modes pass the agreed human-quality checks.

Review `eval_runs/query-review-2026-11-02/audit.json` and its candidate before choosing any replacement for `data/label_queries.json`: 36 encoding repairs, five entity repairs, 32 community-list changes and seven semantic-scope decisions remain subject to user review. Missing original branch-review text (fourteen unidentified minor findings) remains unrecoverable from available artifacts.

### Host backend test record — 2026-11-07

The complete backend pytest suite ran against the project's PostgreSQL container using a uniquely named temporary test database: **1,576 passed** in 15.75 seconds. External Python socket connections were blocked; the local PostgreSQL connection on port 5433 was allowed. HTTP and inference tests used mocks; no real API or model checks were performed. Pytest reported one teardown warning because a test session still held the temporary database open; that test database was subsequently removed. This run does not replace the user-only live inference, GPU, quality, and mode-specific acceptance checks above.
