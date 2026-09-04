# Infrastructure and Deployment

## Docker Services

The system runs as 4 Docker services via `docker compose up`:

```
┌──────────────────────────────────────────────────────────┐
│                     docker compose                        │
├────────────┬────────────┬───────────────┬────────────────┤
│  frontend  │  backend   │    ollama     │      db        │
│  (React)   │  (Django)  │ (params.yaml) │ (PostgreSQL 16)│
│  :3000     │  :8000     │  :11434       │  :5433         │
│            │            │               │                │
│  Vite dev  │  uvicorn   │  auto-pulls   │  pgvector ext  │
│  server    │  ASGI      │  model on     │  exact vector  │
│            │            │  first start  │                │
│  proxies   │  sentence- │               │  catalog +     │
│  /api/*    │  transformers│              │  sessions      │
└────────────┴─────┬──────┴───────┬───────┴────────────────┘
                   │              │
                   │  httpx calls │
                   └──────────────┘
```

### db (PostgreSQL 16 + pgvector)

- **Image**: `pgvector/pgvector:pg16`
- **Port**: 5433 on host (5432 internal, remapped to avoid conflicts with local PostgreSQL)
- **Data**: persisted in `pgdata` Docker volume
- **Health check**: `pg_isready -U recommender`

### ollama (LLM inference)

- **Image**: custom, built from `ollama/Dockerfile`
- **Port**: 11434 (Ollama default)
- **Auto-pull**: custom `entrypoint.sh` starts the Ollama server, waits for
  readiness, reads `llm.model` from the mounted `params.yaml`, and pulls it if
  not cached
- **Model cache**: persisted in `ollama_data` Docker volume (~1.9GB for the
  default qwen2.5:3b)
- **GPU**: reserves an NVIDIA device in `docker-compose.yml`; see Hardware
  Profiles for running without one
- **Health check**: `ollama --version`
- **CRLF fix**: Dockerfile runs `sed -i 's/\r$//'` on entrypoint for Windows compatibility

### backend (Django REST Framework)

- **Image**: custom, built from `backend/Dockerfile` (Python 3.11-slim)
- **Port**: 8000
- **Startup sequence**: `migrate -> import_catalog -> uvicorn`
- **Dependencies**: waits for `db` (healthy) and `ollama` (healthy) before starting
- **Volumes**:
  - `./catalog_okko.parquet:/app/catalog_okko.parquet:ro` (catalog data)
  - `./data:/app/data:ro` (test queries for evaluation)
  - `model_cache:/root/.cache` (sentence-transformers model cache)

### frontend (React + Tailwind)

Not yet implemented. Will serve on port 3000.

## Configuration

### params.yaml (all model and tuning values)

`params.yaml` at the repo root is the single source of truth for every model
name and tuning parameter. It is mounted read-only into the backend
(`/app/params.yaml`) and the ollama container (`/params.yaml`, whose entrypoint
reads `llm.model` from it to decide what to pull). Nothing in it is
overridable by an environment variable -- change the file and restart.

| Section | Covers |
|---------|--------|
| `llm` | Ollama model, base URL, per-call timeouts |
| `embedding` | Embedding model and vector dimensions |
| `retrieval` | Candidate count, RRF constants, lexical search settings |
| `scoring` | Signal weights, neutral score |
| `diversification` | MMR top_n and lambda |
| `reranking` | Cross-encoder model, enable flag, top_k, blend weight |
| `session` | EMA alpha, session TTL |
| `intent` | Genre-match threshold |
| `catalog` | Import and embedding batch sizes |

Changing `embedding.model` or `embedding.dimensions` additionally requires a
schema migration and a full re-run of `generate_embeddings` -- existing vectors
are not convertible.

### Environment variables (secrets and wiring only)

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_PASSWORD` | `changeme` | PostgreSQL password |
| `SECRET_KEY` | insecure dev default | Django secret key |
| `DEBUG` | `0` | Django debug mode |
| `ALLOWED_HOSTS` | `localhost,127.0.0.1` | Comma-separated allowed hosts |
| `CORS_ALLOWED_ORIGINS` | `http://localhost:3000,...` | Comma-separated CORS origins |
| `DATABASE_URL` | compose-provided | Postgres connection string |
| `PARAMS_PATH` | auto-discovered | Override the params.yaml location |

Copy `.env.example` to `.env` and set a real `SECRET_KEY` before first run.

## Hardware Profiles

The committed `params.yaml` targets a discrete NVIDIA GPU with ~11GB of VRAM,
and `docker-compose.yml` reserves the card for Ollama, so the normal command is
all you need:

```bash
docker compose up --build
```

If the NVIDIA driver is missing, Compose fails at startup with a device-driver
error. That is intentional: this profile on CPU is not interactively usable, so
failing loudly beats silently degrading.

VRAM with everything resident:

```
RuadaptQwen3-8B-Hybrid, Q4_K_M   ~5.0 GB
bge-reranker-v2-m3, fp16         ~1.1 GB
mpnet embedder, fp32             ~1.1 GB
CUDA context + activations       ~1.0 GB
                                 --------
                                 ~8.2 GB of 11 GB
```

### Smaller machines

The committed profile does **not** fit an 8GB Apple Silicon Mac, and Docker
Desktop on macOS cannot reach the Apple GPU at all (there is no Metal
passthrough into its Linux VM). Requesting the `nvidia` driver also fails
there, so the `deploy:` block under the `ollama` service has to go.

Alongside that, change `params.yaml`:

| Key | Value | Why |
|-----|-------|-----|
| `llm.model` | `qwen2.5:3b` | ~1.9GB; an 8B model does not fit beside Docker |
| `reranking.enabled` | `false` | bge-v2-m3 on CPU costs ~15s per request |

and run Ollama natively on the host for Metal acceleration, which also keeps
the model out of the Docker VM's memory budget:

```bash
brew install ollama
ollama serve             # in its own terminal
ollama pull qwen2.5:3b
```

Then set `llm.base_url` to `http://host.docker.internal:11434` and start only
the remaining services:

```bash
docker compose up db backend frontend
```

## Management Commands

Run inside the backend container: `docker compose exec backend python manage.py <command>`

| Command | Purpose |
|---------|---------|
| `import_catalog --skip-existing` | Load catalog_okko.parquet into PostgreSQL (idempotent), then rebuild search vectors |
| `generate_embeddings` | Embed all movie descriptions (batch size from params.yaml) |
| `evaluate_scoring` | Run offline evaluation with genre-based metrics |
| `evaluate_scoring --llm-judge` | Run evaluation with LLM graded relevance (slow) |
| `evaluate_scoring --sweep` | Grid search over scoring weight space |
| `evaluate_scoring --sweep --llm-judge` | Grid search with LLM judge (6 configs) |
| `cleanup_sessions` | Delete expired chat sessions (>24h) |

## First Run

```bash
# Start everything (pulls images, builds containers, downloads models)
docker compose up backend

# This automatically:
# 1. Starts PostgreSQL, waits for health check
# 2. Starts Ollama, pulls the model named in params.yaml (first time only)
# 3. Starts backend: runs migrations, imports 18K movies from parquet
# 4. Starts uvicorn on port 8000

# Generate embeddings (separate step, ~5 min on GPU)
docker compose exec backend python manage.py generate_embeddings

# Verify everything works
docker compose exec backend python manage.py shell -c "
from movies.models import Movie
print(f'{Movie.objects.filter(embedding__isnull=False).count()}/{Movie.objects.count()} movies have embeddings')
"
```

## Troubleshooting

**Port 5432 already in use**: Local PostgreSQL running. The db service uses
port 5433 on the host to avoid conflicts.

**Ollama entrypoint fails on Windows**: CRLF line endings. The Dockerfile
runs `sed -i 's/\r$//'` on the entrypoint script. If it still fails,
rebuild: `docker compose build ollama --no-cache`

**"No migrations to apply" but tables don't exist**: Migration files
weren't in the Docker image. Rebuild: `docker compose build backend --no-cache`

**Embedding generation slow**: On CPU, 18K descriptions take ~30-60 minutes.
On GPU, ~5 minutes. Check GPU availability:
`docker compose exec backend python -c "import torch; print(torch.cuda.is_available())"`

**Ollama model not pulling**: Check logs: `docker compose logs ollama`.
The entrypoint waits for the server to be ready before pulling.
