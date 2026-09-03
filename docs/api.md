# API Reference (`backend/movies/views.py`)

The Django REST Framework API layer exposes three HTTP endpoints. All views
are async (Django ASGI) and served by uvicorn. The chat endpoint returns a
two-phase SSE stream: movie results immediately, then a streamed LLM explanation.

Base URL: `http://localhost:8000/api/`

## Endpoints

### `POST /api/chat/`

Accept a natural-language movie query and return personalized recommendations
via Server-Sent Events.

**Request body** (JSON):

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `message` | string | yes | 1-2000 chars, trimmed | Movie query in Russian |
| `session_id` | UUID string | no | valid UUID v4 | Existing session to continue |

**Response**: `text/event-stream` (SSE)

The response is a stream of three event types, always in this order:

```
event: movies
data: {"session_id": "...", "movies": [...], "intent": {...}}

event: token
data: {"text": "..."}
...  (repeated per token)

event: error          (only if explanation generation failed)
data: {"message": "explanation generation failed"}

event: done
data: {}
```

**Phase 1 (instant)**: The `movies` event fires as soon as candidates are scored.
Contains the session UUID (for subsequent requests), the top-5 movie objects,
and the parsed intent.

**Phase 2 (streaming)**: Zero or more `token` events stream the LLM-generated
Russian-language explanation. Tokens arrive as Ollama produces them.

**Terminal**: A `done` event always closes the stream. If explanation generation
failed, an `error` event precedes `done`.

**Movie object shape** (inside the `movies` array):

| Field | Type | Example |
|-------|------|---------|
| `id` | int | `42` |
| `serial_name` | string | `"Интерстеллар"` |
| `genres` | string[] | `["Фантастика", "Драмы"]` |
| `content_type` | string | `"Фильм"` |
| `country` | string[] | `["США"]` |
| `actors` | string[] | `["Мэттью Макконахи"]` |
| `director` | string | `"Кристофер Нолан"` |
| `age_rating` | float\|null | `12.0` |
| `release_date` | string\|null | `"2014-10-26"` |
| `description` | string | `"Когда засуха..."` |
| `url` | string | `"https://okko.tv/movie/interstellar"` |
| `score` | float | `0.8234` |

**Intent object shape**:

| Field | Type | Example |
|-------|------|---------|
| `genres` | string[] | `["Комедии"]` |
| `mood` | string | `"happy"` |
| `themes` | string[] | `["семья"]` |
| `negations` | string[] | `["Ужасы"]` |
| `reference_films` | string[] | `["Один дома"]` |
| `country_exclusions` | string[] | `["США"]` |
| `max_age_rating` | float\|null | `12.0` |
| `min_release_year` | int\|null | `2015` |

`negations` and `country_exclusions` are enforced as SQL `WHERE`/`exclude`
filters in candidate generation, not scoring weights — a movie violating
either never enters the candidate set, regardless of semantic score.
Same for `max_age_rating` and `min_release_year`. Movies with a null
`age_rating` or `release_date` are never excluded by these filters (treated
as "unknown, don't filter" rather than "fails the constraint").

**Error responses** (JSON, not SSE):

| Status | Body | Cause |
|--------|------|-------|
| 400 | `{"error": "message is required"}` | Empty or missing message |
| 400 | `{"error": "message too long (max 2000 chars)"}` | Message exceeds 2000 chars |

**Example** (curl):

```bash
curl -N -X POST http://localhost:8000/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{"message": "хочу комедию про семью"}'
```

**Example with session continuation**:

```bash
curl -N -X POST http://localhost:8000/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{"message": "а что-нибудь поновее?", "session_id": "a1b2c3d4-..."}'
```

---

### `GET /api/sessions/<uuid:session_id>/`

Retrieve the conversation history and learned preferences for an existing session.
Requires the `X-Session-Token` header for authentication.

**Path parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | UUID | The session UUID returned by the chat endpoint |

**Required headers**:

| Header | Description |
|--------|-------------|
| `X-Session-Token` | The `session_token` value returned in the SSE `movies` or `session` event |

**Response** (JSON, 200):

```json
{
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "history": [
    {
      "role": "user",
      "content": "хочу комедию",
      "movies": ["Один дома", "Ирония судьбы"]
    }
  ],
  "turn_count": 1,
  "preferences": {
    "liked_genres": ["Комедии"],
    "disliked_genres": [],
    "themes": [],
    "reference_films": []
  }
}
```

**Error responses**:

| Status | Body | Cause |
|--------|------|-------|
| 401 | `{"error": "session token required"}` | Missing `X-Session-Token` header |
| 403 | `{"error": "invalid session token"}` | Token does not match the session |
| 404 | `{"error": "session not found"}` | UUID does not match any session |

---

### `GET /api/health/`

Returns service health and catalog size.

**Response** (JSON, 200):

```json
{
  "status": "healthy",
  "timestamp": "2024-06-23T15:30:00.000000",
  "catalog_size": 18130
}
```

---

## Request Pipeline

What happens inside `POST /api/chat/`:

```
POST /api/chat/ {"message": "...", "session_id": "..."}
  |
  ├── validate message (non-empty, <= 2000 chars)
  ├── get or create session (expired sessions are replaced)
  |
  ├──[PARALLEL via asyncio.gather]──┐
  │   aparse_intent(message)        │  encode_query(message)
  │   (Ollama LLM, ~2s)            │  (sentence-transformers, ~0.3s)
  └──────────────┬──────────────────┘
                 v
  generate_candidates(embedding, intent)     ← pgvector HNSW ANN search
  hybrid_score() per candidate               ← semantic + metadata + session
  mmr_diversify(scored, top_n=5)             ← diversity reranking
                 |
  _save_session()                            ← atomic update with SELECT FOR UPDATE
                 |
  SSE Phase 1:   event: movies  {results + session_id + intent}
  SSE Phase 2:   event: token   {text} ... (streamed from Ollama)
  SSE Terminal:  event: done    {}
```

## Session Lifecycle

Sessions auto-expire after 24 hours (`ChatSession.is_expired()`). On each turn:

1. The 768-dim preference vector is updated via exponential moving average
   (alpha=0.7, configurable via `SESSION_ALPHA` env var).
2. Explicit preferences (liked genres, disliked genres, themes, reference films)
   are accumulated from the parsed intent.
3. The conversation history is appended with the query and recommended movie titles.
4. `turn_count` increments.

Session updates use `SELECT FOR UPDATE` inside a transaction to prevent
lost updates from concurrent requests on the same session.

If a `session_id` is not provided or doesn't match an existing non-expired
session, a new session is created and its UUID returned in the first SSE event.

## Configuration

All settings are in `backend/recommender/settings.py`, overridable via environment variables:

| Env var | Default | Effect |
|---------|---------|--------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Model for intent parsing and explanations |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | Embedding model |
| `SCORE_WEIGHT_SEMANTIC` | `0.4` | Weight for cosine similarity score |
| `SCORE_WEIGHT_METADATA` | `0.3` | Weight for genre overlap score |
| `SCORE_WEIGHT_SESSION` | `0.3` | Weight for session preference score |
| `SESSION_ALPHA` | `0.7` | EMA alpha for preference vector updates |
| `GENRE_MATCH_THRESHOLD` | `0.5` | Min cosine similarity for genre normalization |

## Threading Model

The API runs on Django ASGI via uvicorn. Key threading decisions:

- **Intent parsing** (`aparse_intent`): fully async via `httpx.AsyncClient`.
- **Embedding encoding** (`encode_query`): wrapped in `sync_to_async(thread_sensitive=True)`.
  This serializes embedding calls on the main thread because sentence-transformers
  uses shared GPU state that is not thread-safe.
- **Candidate generation + scoring** (`_generate_and_score`): wrapped in
  `sync_to_async(thread_sensitive=False)`. Runs in the general thread pool because
  the Django ORM and numpy computations are thread-safe.
- **Explanation streaming** (`astream_explanation`): fully async generator via
  `httpx.AsyncClient.stream()`.
- **Session save** (`_save_session`): `sync_to_async(thread_sensitive=False)` with
  `transaction.atomic()` and `select_for_update()`.

## Related

- [Frontend](frontend.md) for the React chat UI that consumes these endpoints
- [Core ML Pipeline](core.md) for scoring weights, MMR, and embedding details
- [Data Models](models.md) for Movie and ChatSession schemas
- [Infrastructure](infrastructure.md) for Docker services and deployment
- [ML Decisions](ML.md) for evaluation methodology and parameter tuning
