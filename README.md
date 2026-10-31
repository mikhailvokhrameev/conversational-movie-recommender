# Hybrid Movie Recommender

## What is it

A conversational movie recommendation system built on a ~50K-movie TMDB catalog. Users describe what they want to watch in natural language, and the system returns personalized picks with real-time streaming explanations of why each movie fits.

The chat understands context: follow-up questions about recommended movies, preference refinements ("something funnier?"), and general conversation are handled differently from new search queries. Preferences are learned across the conversation via an embedding-based preference vector.

## Motivation

Most movie recommendation systems are either keyword-based search (fragile, no understanding of intent) or collaborative filtering (cold-start problem, no explainability). This project combines multiple signals into a hybrid approach:

- **Semantic search** via a BGE-M3 embedder + pgvector exact cosine distance
- **Lexical search** via a Postgres full-text channel over title/keywords, fused with the semantic channel by Reciprocal Rank Fusion -- catches named titles that embeddings miss
- **Reranking** over the fused candidate pool (local BGE or cloud Voyage)
- **Metadata matching** via LLM-powered intent parsing that extracts genres, mood, themes, negations, and hard filters (rating floor, release year, runtime, language, country) from natural language
- **Session-based preference learning** via exponential moving average on the user's query embeddings

The shared LLM layer handles intent parsing, classification and explanations through local Ollama or OpenRouter adapters. The same retrieval/scoring pipeline runs in both modes; cloud mode includes catalog embeddings and reranking without GPU or local ML loads.

## Architecture

```text
React/Vite :3000 -> Django/uvicorn :8000 -> PostgreSQL/pgvector :5433 (host)
                            |
                   cached adapter factory
                    /
    local: Ollama + BGE models      cloud: OpenRouter (all three components)
```


**Request flow:**

```
User message
  │
  ├── Classify + parse intent (selected LLM adapter, JSON schema, one call) → category + genres/mood/themes/filters
  │
  ├── [new_search / refinement]
  │     ├── Encode query (BGE-M3)
  │     ├── Candidate generation (hard filters, then semantic + lexical channels fused by RRF, top 100)
  │     ├── Hybrid scoring (semantic 0.35 + metadata 0.25 + session 0.25 + popularity 0.15)
  │     ├── Reranking (local BGE or cloud Voyage, top 50)
  │     ├── MMR diversification (top 5)
  │     └── SSE stream: movie cards (instant) + explanation (token by token)
  │
  ├── [follow_up] → LLM response with last movies as context (text only)
  └── [general_chat] → LLM response, stays in movie assistant character
```

## Stack

| Layer | Technology |
|-------|-----------|
| LLM | Local Ollama `qwen3.5:9b-q4_K_M` or OpenRouter `qwen/qwen3.5-9b` |
| Embeddings | BAAI/bge-m3 (1024-dim) |
| Reranker | Local BAAI/bge-reranker-v2-m3 or cloud voyageai/rerank-3-lite |
| Vector DB | PostgreSQL 16 + pgvector (exact cosine distance) + full-text (tsvector/RRF) |
| Backend | Django 4.2, async views, uvicorn ASGI, httpx |
| Frontend | React 19, Vite, Tailwind CSS 4, OKLCH dark/light theme |
| Infrastructure | Docker Compose (4 local services; 3 cloud services) |
| Tests | pytest, pytest-django, pytest-asyncio |

## How to run

**Requirements:** Docker and Docker Compose; NVIDIA GPU for local mode, or an OpenRouter API key for cloud mode. Real hardware/API acceptance remains pending; see [deployment and manual acceptance](docs/infrastructure.md).

```bash
git clone https://github.com/mikhailvokhrameev/hybrid-movie-recommender.git
cd hybrid-movie-recommender

# Create .env with a real secret key
cp .env.example .env
# Edit .env and replace SECRET_KEY with a random string

# Start all services
docker compose up --build
```

For cloud instead run `docker compose -f docker-compose.yml -f docker-compose.openrouter.yml up --build` with `OPENROUTER_API_KEY` configured. Before switching, run the current mode’s Compose command with `down` without `-v` to preserve volumes. Local startup pulls `qwen3.5:9b-q4_K_M`; its VRAM fit and compatibility with the pinned Ollama image need user validation. The backend will run migrations and import the TMDB movie catalog (~50K movies -- the first import + embedding pass takes a while).

Once everything is up, open http://localhost:3000 and start chatting.

**Useful commands:**

```bash
# Run tests
docker compose exec backend pytest -v

# Check service health
curl http://localhost:8000/api/health/

# View session history (requires session token from SSE response)
curl -H "X-Session-Token: <token>" http://localhost:3000/api/sessions/<uuid>/

# Generate missing catalog embeddings (explicit separate step)
docker compose exec backend python manage.py generate_embeddings

# Evaluate against the golden set (test split; run record in eval_runs/)
docker compose exec backend python manage.py evaluate_scoring

# Tune scoring weights on the dev split
docker compose exec backend python manage.py evaluate_scoring --sweep
```

## Project structure

```
backend/
├── core/                       # ML pipeline
│   ├── llm_client.py            # LLM: classify+parse intent, explain, chat
│   ├── embedding_service.py        # BGE-M3 wrapper
│   ├── candidate_generation.py     # hard filters + semantic/lexical RRF fusion
│   ├── reranking.py                # cross-encoder reranking stage
│   ├── scoring.py                  # hybrid scoring + MMR diversification
│   ├── session_manager.py          # EMA preference vector
│   ├── pipeline.py                 # retrieval pipeline shared by chat view and eval
│   ├── golden_set.py               # golden-set loading + reference resolution
│   └── evaluation.py              # metrics, LLM-as-judge
├── movies/
│   ├── models.py                   # Movie + ChatSession (pgvector fields)
│   ├── views.py                    # async API: chat, sessions, health
│   ├── search_index.py             # full-text search_vector maintenance
│   ├── urls.py                     # /api/chat/, /api/sessions/, /api/health/
│   └── management/commands/        # import_catalog, generate_embeddings, evaluate_scoring
└── recommender/
    └── settings.py                 # Django config (env vars + params.yaml)

frontend/src/
├── components/                 # ChatMessage, MovieCard, Header, ThemeToggle
├── hooks/                      # useChat (SSE streaming), useTheme (dark/light)
└── index.css                   # OKLCH design tokens

docs/                           # API reference, ML decisions, infrastructure, datasheet, ADRs
```

## Data attribution

Movie data and poster images come from [TMDB](https://www.themoviedb.org/). This product uses the TMDB API but is not endorsed or certified by TMDB. Posters are loaded in the browser from TMDB's image CDN (`https://image.tmdb.org/t/p/`), not stored by this project. See the [catalog datasheet](docs/datasheet.md) for what the catalog contains and its known limits.

## Documentation

- [API Reference](docs/api.md)
- [Core ML Pipeline](docs/core.md)
- [ML Decisions](docs/ML.md)
- [Data Models](docs/models.md)
- [Catalog Datasheet](docs/datasheet.md)
- [Architecture Decision Records](docs/adr/README.md)
- [Frontend](docs/frontend.md)
- [Infrastructure](docs/infrastructure.md)
