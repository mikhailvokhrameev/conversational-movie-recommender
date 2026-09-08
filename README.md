# Hybrid Movie Recommender

## What is it

A conversational movie recommendation system built on a ~50K-movie TMDB catalog. Users describe what they want to watch in natural language, and the system returns personalized picks with real-time streaming explanations of why each movie fits.

The chat understands context: follow-up questions about recommended movies, preference refinements ("something funnier?"), and general conversation are handled differently from new search queries. Preferences are learned across the conversation via an embedding-based preference vector.

## Motivation

Most movie recommendation systems are either keyword-based search (fragile, no understanding of intent) or collaborative filtering (cold-start problem, no explainability). This project combines multiple signals into a hybrid approach:

- **Semantic search** via a BGE-M3 embedder + pgvector exact cosine distance
- **Lexical search** via a Postgres full-text channel over title/keywords, fused with the semantic channel by Reciprocal Rank Fusion -- catches named titles that embeddings miss
- **Cross-encoder reranking** (bge-reranker-v2-m3) over the fused candidate pool
- **Metadata matching** via LLM-powered intent parsing that extracts genres, mood, themes, negations, and hard filters (rating floor, release year, runtime, language, country) from natural language
- **Session-based preference learning** via exponential moving average on the user's query embeddings

The LLM layer (Ollama with qwen3:8b, running locally on GPU) handles intent parsing, message classification, and explanation generation. Everything runs locally via Docker, no external API calls.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Docker Compose                          │
│                                                              │
│  ┌───────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │ PostgreSQL │  │    Ollama    │  │     Frontend       │   │
│  │ + pgvector │  │   qwen3:8b   │  │  React + Vite      │   │
│  │   :5432    │  │   :11434    │  │     :3000           │   │
│  └─────┬──────┘  └──────┬──────┘  └──────────┬─────────┘   │
│        │                │                     │ proxy /api   │
│        │         ┌──────┴──────────┐          │             │
│        └─────────┤    Backend      ├──────────┘             │
│                  │ Django + uvicorn │                        │
│                  │     :8000       │                         │
│                  └─────────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

**Request flow:**

```
User message
  │
  ├── Classify + parse intent (Ollama, JSON mode, one call) → category + genres/mood/themes/filters
  │
  ├── [new_search / refinement]
  │     ├── Encode query (BGE-M3)
  │     ├── Candidate generation (hard filters, then semantic + lexical channels fused by RRF, top 100)
  │     ├── Hybrid scoring (semantic 0.35 + metadata 0.25 + session 0.25 + popularity 0.15)
  │     ├── Cross-encoder reranking (bge-reranker-v2-m3, top 50)
  │     ├── MMR diversification (top 5)
  │     └── SSE stream: movie cards (instant) + explanation (token by token)
  │
  ├── [follow_up] → LLM response with last movies as context (text only)
  └── [general_chat] → LLM response, stays in movie assistant character
```

## Stack

| Layer | Technology |
|-------|-----------|
| LLM | Ollama + qwen3:8b (local, GPU-accelerated) |
| Embeddings | BAAI/bge-m3 (1024-dim) |
| Reranker | BAAI/bge-reranker-v2-m3 (cross-encoder) |
| Vector DB | PostgreSQL 16 + pgvector (exact cosine distance) + full-text (tsvector/RRF) |
| Backend | Django 4.2, async views, uvicorn ASGI, httpx |
| Frontend | React 19, Vite, Tailwind CSS 4, OKLCH dark/light theme |
| Infrastructure | Docker Compose (4 services) |
| Tests | pytest, pytest-django, pytest-asyncio |

## How to run

**Requirements:** Docker, Docker Compose, NVIDIA GPU (for Ollama).

```bash
git clone https://github.com/mikhailvokhrameev/hybrid-movie-recommender.git
cd hybrid-movie-recommender

# Create .env with a real secret key
cp .env.example .env
# Edit .env and replace SECRET_KEY with a random string

# Start all services
docker compose up --build
```

On first start, Ollama will automatically pull the qwen3:8b model (~5.0 GB). The backend will run migrations and import the TMDB movie catalog (~50K movies -- the first import + embedding pass takes a while).

Once everything is up, open http://localhost:3000 and start chatting.

**Useful commands:**

```bash
# Run tests
docker compose exec backend pytest -v

# Check service health
curl http://localhost:8000/api/health/

# View session history (requires session token from SSE response)
curl -H "X-Session-Token: <token>" http://localhost:3000/api/sessions/<uuid>/

# Generate embeddings for the catalog (runs on first import)
docker compose exec backend python manage.py generate_embeddings

# Evaluate scoring quality
docker compose exec backend python manage.py evaluate_scoring --sweep
```

## Project structure

```
backend/
├── core/                       # ML pipeline
│   ├── ollama_client.py            # LLM: classify+parse intent, explain, chat
│   ├── embedding_service.py        # BGE-M3 wrapper
│   ├── candidate_generation.py     # hard filters + semantic/lexical RRF fusion
│   ├── reranking.py                # cross-encoder reranking stage
│   ├── scoring.py                  # hybrid scoring + MMR diversification
│   ├── session_manager.py          # EMA preference vector
│   └── evaluation.py              # LLM-as-judge, metrics
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

docs/                           # API reference, ML decisions, infrastructure
```

## Documentation

- [API Reference](docs/api.md)
- [Core ML Pipeline](docs/core.md)
- [ML Decisions](docs/ML.md)
- [Data Models](docs/models.md)
- [Frontend](docs/frontend.md)
- [Infrastructure](docs/infrastructure.md)
