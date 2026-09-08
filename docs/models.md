# Data Models Reference

## Movie

Represents a movie from the TMDB catalog.

**Table**: `movies_movie`

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | BigAutoField | PK | Auto-generated primary key |
| `tmdb_id` | IntegerField | unique | TMDB's own movie id |
| `serial_name` | CharField(500) | indexed | Display title (TMDB's `title`) |
| `original_title` | CharField(500) | blank | Title in the movie's original language |
| `genres` | JSONField | default=[] | List of genre strings, e.g. `["Comedy", "Drama"]` (see Genre Taxonomy) |
| `country` | JSONField | default=[] | List of production countries |
| `original_language` | CharField(10) | blank | ISO 639-1 code, e.g. `en`, `ko` |
| `keywords` | JSONField | default=[] | Curated theme/franchise/character tags from TMDB |
| `release_date` | DateField | nullable | Original release date |
| `description` | TextField | blank | English-language synopsis (TMDB `overview`) |
| `runtime` | IntegerField | nullable | Runtime in minutes |
| `popularity` | FloatField | default=0.0 | TMDB's popularity metric (unbounded, non-linear) |
| `vote_average` | FloatField | default=0.0 | TMDB user rating, 0-10 |
| `vote_count` | IntegerField | default=0 | Number of TMDB votes (import floor: see Catalog Import below) |
| `poster_path` | CharField(500) | nullable | TMDB poster path; combine with `catalog.poster_base_url`/`poster_size` from `params.yaml` for a full image URL |
| `embedding` | VectorField(1024) | nullable | Pre-computed description embedding (BGE-M3) |
| `search_vector` | SearchVectorField | nullable, GIN | Full-text vector over title/original_title (A), keywords (B) |

**Indexes**:
- B-tree on `serial_name` (Django `db_index=True`)
- No ANN index on `embedding` -- exact cosine search (see ML.md for why)
- GIN on `search_vector` (`movie_search_vector_gin`) for full-text matching
- Unique constraint on `tmdb_id`

`search_vector` is not maintained on save. The catalog is bulk-imported, so
`movies.search_index.refresh_search_vectors()` rebuilds it in a single UPDATE
after import; `import_catalog` calls it automatically.

**Data source**: `TMDB Movie Dataset v11.csv`, filtered at import time to
`status=Released`, non-adult, has overview, has poster, `vote_count >= 20`
(~50K rows imported from ~1.49M raw). Loaded via `manage.py import_catalog`.
Embeddings generated via `manage.py generate_embeddings`.

### Genre Taxonomy

TMDB's standard 19-genre list (used both for the catalog's `genres` field and
as the LLM's `ALLOWED GENRES` list in intent parsing):

Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Family,
Fantasy, History, Horror, Music, Mystery, Romance, Science Fiction, TV Movie,
Thriller, War, Western

## ChatSession

Represents a single conversation session with a user. Stores the evolving
preference profile for session-based recommendation learning.

**Table**: `movies_chatsession`

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | BigAutoField | PK | Auto-generated primary key |
| `session_id` | UUIDField | unique, indexed | Frontend-generated session identifier |
| `session_token` | CharField(64) | indexed | Cryptographic token for session auth (`secrets.token_urlsafe(32)`) |
| `preference_vector` | VectorField(1024) | nullable | EMA-updated preference embedding |
| `preferences` | JSONField | default={} | Explicit preferences (liked/disliked genres, themes) |
| `history` | JSONField | default=[] | Conversation message history |
| `turn_count` | IntegerField | default=0 | Number of conversation turns |
| `created_at` | DateTimeField | auto | Session creation time |
| `updated_at` | DateTimeField | auto | Last interaction time |

**Indexes**:
- Unique on `session_id`
- B-tree on `updated_at` (for cleanup queries)

**Session lifecycle**:
1. Frontend generates UUID on first load, stores in localStorage
2. Backend creates ChatSession via `get_or_create` on first request
3. Each query updates `preference_vector` (EMA), `preferences` (explicit),
   and increments `turn_count`
4. `manage.py cleanup_sessions` deletes sessions where
   `created_at < now() - 24 hours`

### Preferences Structure

The `preferences` JSONField stores:

```json
{
  "liked_genres": ["Comedy", "Adventure"],
  "disliked_genres": ["Horror"],
  "themes": ["space", "travel"],
  "reference_films": ["Interstellar"]
}
```

Updated by `session_manager.track_explicit_preferences()` after each
intent parsing. Uses append-if-not-present semantics (no duplicates).

## Database Setup

PostgreSQL 16 with pgvector extension. The initial migration
(`0001_initial.py`) runs `CREATE EXTENSION IF NOT EXISTS vector` before
creating tables.

Connection configured via `DATABASE_URL` environment variable:
`postgres://recommender:PASSWORD@db:5432/recommender`

## Related

- [API Reference](api.md) for HTTP endpoints and session lifecycle
- [Core ML Pipeline](core.md) for how models are used in scoring
- [Infrastructure](infrastructure.md) for Docker and database setup
