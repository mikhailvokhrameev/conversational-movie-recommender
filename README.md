# Hybrid Movie Recommender

## What is it

A conversational movie recommendation system for the Okko streaming platform. Users describe what they want to watch in Russian natural language, and the system returns personalized picks with real-time streaming explanations of why each movie fits.

The chat understands context: follow-up questions about recommended movies, preference refinements ("а повеселее?"), and general conversation are handled differently from new search queries. Preferences are learned across the conversation via an embedding-based preference vector.

## Motivation

Most movie recommendation systems are either keyword-based search (fragile, no understanding of intent) or collaborative filtering (cold-start problem, no explainability). This project combines three signals into a hybrid approach:

- **Semantic search** via sentence-transformers embeddings + pgvector HNSW index for fast approximate nearest neighbor retrieval
- **Metadata matching** via LLM-powered intent parsing that extracts genres, mood, themes, and negations from natural language
- **Session-based preference learning** via exponential moving average on the user's query embeddings

The LLM layer (Ollama with qwen2.5:7b, running locally on GPU) handles intent parsing, message classification, and Russian-language explanation generation. Everything runs locally via Docker, no external API calls.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Docker Compose                          │
│                                                              │
│  ┌───────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │ PostgreSQL │  │    Ollama    │  │     Frontend       │   │
│  │ + pgvector │  │ qwen2.5:7b  │  │  React + Vite      │   │
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
  ├── Classify message (Ollama) → new_search / follow_up / refinement / general_chat
  │
  ├── [new_search / refinement]
  │     ├── Parse intent (Ollama, JSON mode) ──┐
  │     │                                      ├── parallel
  │     ├── Encode query (sentence-transformers) ──┘
  │     ├── Candidate generation (pgvector HNSW, top 100)
  │     ├── Hybrid scoring (semantic 0.4 + metadata 0.3 + session 0.3)
  │     ├── MMR diversification (top 5)
  │     └── SSE stream: movie cards (instant) + explanation (token by token)
  │
  ├── [follow_up] → LLM response with last movies as context (text only)
  └── [general_chat] → LLM response, stays in movie assistant character
```

## Stack

| Layer | Technology |
|-------|-----------|
| LLM | Ollama + qwen2.5:7b (local, GPU-accelerated) |
| Embeddings | sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (768-dim) |
| Vector DB | PostgreSQL 16 + pgvector (HNSW index, cosine distance) |
| Backend | Django 4.2, async views, uvicorn ASGI, httpx |
| Frontend | React 18, Vite, Tailwind CSS 4, OKLCH dark/light theme |
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

On first start, Ollama will automatically pull the qwen2.5:7b model (~4.7 GB). The backend will run migrations and import the movie catalog.

Once everything is up, open http://localhost:3000 and start chatting in Russian.

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
│   ├── ollama_client.py            # LLM: intent parse, classify, explain, chat
│   ├── embedding_service.py        # sentence-transformers wrapper
│   ├── candidate_generation.py     # pgvector ANN search
│   ├── scoring.py                  # hybrid scoring + MMR diversification
│   ├── session_manager.py          # EMA preference vector
│   └── evaluation.py              # LLM-as-judge, metrics
├── movies/
│   ├── models.py                   # Movie + ChatSession (pgvector fields)
│   ├── views.py                    # async API: chat, sessions, health
│   └── urls.py                     # /api/chat/, /api/sessions/, /api/health/
└── recommender/
    └── settings.py                 # Django config (env vars)

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
