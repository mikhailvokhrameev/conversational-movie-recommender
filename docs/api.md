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
| `message` | string | yes | 1-2000 chars, trimmed | Movie query in natural language |
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
explanation. Tokens arrive as Ollama produces them.

**Terminal**: A `done` event always closes the stream. If explanation generation
failed, an `error` event precedes `done`.

**Movie object shape** (inside the `movies` array):

| Field | Type | Example |
|-------|------|---------|
| `id` | int | `42` |
| `tmdb_id` | int | `157336` |
| `serial_name` | string | `"Interstellar"` |
| `original_title` | string | `"Interstellar"` |
| `genres` | string[] | `["Science Fiction", "Drama"]` |
| `country` | string[] | `["US", "GB"]` |
| `release_date` | string\|null | `"2014-10-26"` |
| `description` | string | `"The adventures of a group of explorers..."` |
| `runtime` | int\|null | `169` |
| `vote_average` | float | `8.4` |
| `poster_url` | string\|null | `"https://image.tmdb.org/t/p/w500/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg"` |
| `score` | float | `0.8234` |

**Intent object shape**:

| Field | Type | Example |
|-------|------|---------|
| `genres` | string[] | `["Comedy"]` |
| `mood` | string | `"happy"` |
| `themes` | string[] | `["family"]` |
| `negations` | string[] | `["Horror"]` |
| `reference_films` | string[] | `["Home Alone"]` |
| `country_exclusions` | string[] | `["US"]` |
| `country_inclusions` | string[] | `[]` |
| `min_vote_average` | float\|null | `7.0` |
| `min_release_year` | int\|null | `2015` |
| `max_release_year` | int\|null | `null` |
| `min_runtime` | int\|null | `null` |
| `max_runtime` | int\|null | `90` |
| `original_languages` | string[] | `["ko"]` |

`negations`, `country_exclusions`/`country_inclusions`, `min_vote_average`,
`min_release_year`/`max_release_year`, `min_runtime`/`max_runtime`, and
`original_languages` are all enforced as SQL `WHERE`/`exclude` filters in
candidate generation, not scoring weights — a movie violating any of them
never enters the candidate set, regardless of semantic score. Movies with a
null `vote_average`, `release_date`, or `runtime` are never excluded by these
filters (treated as "unknown, don't filter" rather than "fails the
constraint") — `original_languages` is the one exception, since
`original_language` is always known for an imported row.

**Error responses** (JSON, not SSE):

| Status | Body | Cause |
|--------|------|-------|
| 400 | `{"error": "message is required"}` | Empty or missing message |
| 400 | `{"error": "message too long (max 2000 chars)"}` | Message exceeds 2000 chars |

**Example** (curl):

```bash
curl -N -X POST http://localhost:8000/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{"message": "I want a comedy about family"}'
```

**Example with session continuation**:

```bash
curl -N -X POST http://localhost:8000/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{"message": "something newer?", "session_id": "a1b2c3d4-..."}'
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
      "content": "I want a comedy",
      "movies": ["Home Alone", "Groundhog Day"]
    }
  ],
  "turn_count": 1,
  "preferences": {
    "liked_genres": ["Comedy"],
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
  "catalog_size": 50000
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
  aclassify_and_parse(message)                ← single Ollama call: category +
  (Ollama LLM, ~2s, Pydantic-validated,          filters + semantic_query
   1 retry-with-repair on invalid JSON)
                 |
  encode_query(intent.semantic_query)        ← BGE-M3, ~0.3s
                 v
  generate_candidates(embedding, intent)     ← hard filters, then semantic + lexical
                                               channels fused by RRF
  score_candidates(candidates, ...)          ← semantic + metadata + session + popularity,
                                               each normalized across the set
  rerank_candidates(query, scored)           ← cross-encoder, on by default, top_k=50
  mmr_diversify(scored, top_n=5)             ← diversity selection
                 |
  _save_session()                            ← atomic update with SELECT FOR UPDATE
                 |
  SSE Phase 1:   event: movies  {results + session_id + intent}
  SSE Phase 2:   event: token   {text} ... (streamed from Ollama)
  SSE Terminal:  event: done    {}
```

## Session Lifecycle

Sessions auto-expire after `session.ttl_hours` (default 24) via
`ChatSession.is_expired()`. On each turn:

1. The preference vector (`embedding.dimensions` wide) is updated via an
   exponential moving average
   (alpha from `session.alpha` in params.yaml, default 0.7; turns classified
   `refinement` use the higher `session.alpha_refinement`, default 0.92, so
   an explicit correction dominates the vector instead of blending).
2. Explicit preferences (liked genres, disliked genres, themes, reference films)
   are accumulated from the parsed intent.
3. The conversation history is appended with the query and recommended movie titles.
4. `turn_count` increments.

Session updates use `SELECT FOR UPDATE` inside a transaction to prevent
lost updates from concurrent requests on the same session.

If a `session_id` is not provided or doesn't match an existing non-expired
session, a new session is created and its UUID returned in the first SSE event.

## Configuration

Every model name and tuning value lives in `params.yaml` at the repo root,
which `backend/recommender/settings.py` loads at startup. It is the only place
these can be changed -- no environment variable overrides them.

| params.yaml path | Default | Effect |
|------------------|---------|--------|
| `llm.base_url` | `http://ollama:11434` | Ollama server URL |
| `llm.model` | `qwen3:8b` | Model for classification, intent parsing, explanations |
| `llm.thinking` | `false` | Suppress hybrid-reasoning `<think>` spans |
| `llm.timeout_seconds.*` | 45 / 60 / 120 | Per-call timeouts (classify_parse / intent / explanation) |
| `embedding.model` | `BAAI/bge-m3` | Embedding model |
| `embedding.dimensions` | `1024` | Vector width (changing needs a migration + re-embed) |
| `retrieval.candidate_count` | `100` | Candidates per retrieval channel |
| `retrieval.rrf_k` | `60` | RRF rank-smoothing constant |
| `retrieval.rrf_weights.*` | 1.0 / 0.7 | Channel weights (semantic / lexical) |
| `scoring.weights.*` | 0.35 / 0.25 / 0.25 / 0.15 | Signal weights (semantic / metadata / session / popularity) |
| `diversification.top_n` | `5` | Results returned |
| `diversification.lambda` | `0.7` | MMR relevance-vs-diversity tradeoff |
| `reranking.enabled` | `true` | Cross-encoder reranking on/off |
| `reranking.model` | `BAAI/bge-reranker-v2-m3` | Cross-encoder model |
| `reranking.device` | `auto` | `auto` / `cuda` / `cpu` (see ML.md before pinning) |
| `reranking.top_k` | `50` | Candidates sent to the cross-encoder |
| `reranking.weight` | `0.5` | Share of final score from the cross-encoder |
| `reranking.description_chars` | `400` | Description chars sent per candidate to the cross-encoder |
| `session.alpha` | `0.7` | EMA alpha for preference vector updates |
| `session.alpha_refinement` | `0.92` | EMA alpha used instead, on `refinement` turns |
| `session.ttl_hours` | `24` | Session expiry |
| `intent.genre_match_threshold` | `0.5` | Min cosine similarity for genre normalization |

Environment variables are reserved for secrets and deployment wiring
(`SECRET_KEY`, `DB_PASSWORD`, `DEBUG`, `ALLOWED_HOSTS`, `CORS_ALLOWED_ORIGINS`,
`DATABASE_URL`). See [Infrastructure](infrastructure.md#configuration).

## Threading Model

The API runs on Django ASGI via uvicorn. Key threading decisions:

- **Classification + intent parsing** (`aclassify_and_parse`): fully async via `httpx.AsyncClient`,
  one combined call validated against the `MessageIntent` Pydantic schema (with a
  repair retry on invalid JSON) instead of two separate classify/parse calls.
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
