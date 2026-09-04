# Infrastructure and Deployment

## Docker Services

The system runs as 4 Docker services via `docker compose up`:

```
┌──────────────────────────────────────────────────────────┐
│                     docker compose                        │
├────────────┬────────────┬───────────────┬────────────────┤
│  frontend  │  backend   │    ollama     │      db        │
│  (React)   │  (Django)  │  (qwen2.5:7b) │ (PostgreSQL 16)│
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
- **Auto-pull**: custom `entrypoint.sh` starts Ollama server, waits for readiness,
  pulls the configured model (`OLLAMA_MODEL`, default `qwen2.5:7b`) if not cached
- **Model cache**: persisted in `ollama_data` Docker volume (~4GB for qwen2.5:7b)
- **GPU**: uses NVIDIA GPU if available via Docker `deploy.resources.reservations`
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

## Environment Variables

All configuration via environment variables, with defaults in `docker-compose.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_PASSWORD` | `changeme` | PostgreSQL password |
| `SECRET_KEY` | `change-in-production` | Django secret key |
| `DEBUG` | `0` | Django debug mode |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Ollama model for intent parsing and explanations |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | Embedding model name |
| `SCORE_WEIGHT_SEMANTIC` | `0.4` | Semantic similarity weight in hybrid scoring |
| `SCORE_WEIGHT_METADATA` | `0.3` | Metadata match weight |
| `SCORE_WEIGHT_SESSION` | `0.3` | Session preference weight |
| `SESSION_ALPHA` | `0.7` | EMA decay factor for session preference vector |
| `GENRE_MATCH_THRESHOLD` | `0.5` | Cosine similarity threshold for genre normalization |

Copy `.env.example` to `.env` and customize before first run.

## Management Commands

Run inside the backend container: `docker compose exec backend python manage.py <command>`

| Command | Purpose |
|---------|---------|
| `import_catalog --skip-existing` | Load catalog_okko.parquet into PostgreSQL (idempotent) |
| `generate_embeddings --batch-size 64` | Embed all movie descriptions via sentence-transformers |
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
# 2. Starts Ollama, pulls qwen2.5:7b model (~4GB, first time only)
# 3. Starts backend: runs migrations, imports 18K movies from parquet
# 4. Starts uvicorn on port 8000

# Generate embeddings (separate step, ~5 min on GPU)
docker compose exec backend python manage.py generate_embeddings --batch-size 64

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
