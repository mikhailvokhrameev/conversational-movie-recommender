# Data Models Reference

## Movie

Represents a movie, series, or multi-episode film from the Okko catalog.

**Table**: `movies_movie`

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | BigAutoField | PK | Auto-generated primary key |
| `serial_name` | CharField(500) | indexed | Title in Russian |
| `genres` | JSONField | default=[] | List of genre strings, e.g. `["Комедии", "Драмы"]` |
| `content_type` | CharField(50) | | One of: `Фильм`, `Сериал`, `Многосерийный фильм` |
| `country` | JSONField | default=[] | List of countries, e.g. `["Россия", "США"]` |
| `actors` | JSONField | default=[] | List of actor names (max 4 per movie in catalog) |
| `director` | CharField(255) | blank | Director name |
| `age_rating` | FloatField | nullable | Age restriction (6, 12, 16, 18, or null) |
| `studio_name` | CharField(500) | blank | Production studio |
| `release_date` | DateField | nullable | Original release date |
| `description` | TextField | blank | Russian-language synopsis (avg 590 chars) |
| `url` | URLField(500) | unique | Okko catalog URL |
| `embedding` | VectorField(768) | nullable | Pre-computed description embedding |
| `search_vector` | SearchVectorField | nullable, GIN | Full-text vector over title (A), director/actors (B) |

**Indexes**:
- B-tree on `serial_name` (Django `db_index=True`)
- No ANN index on `embedding` -- exact cosine search (see ML.md for why)
- GIN on `search_vector` (`movie_search_vector_gin`) for full-text matching

`search_vector` is not maintained on save. The catalog is bulk-imported, so
`movies.search_index.refresh_search_vectors()` rebuilds it in a single UPDATE
after import; `import_catalog` calls it automatically.
- Unique constraint on `url`

**Data source**: `catalog_okko.parquet` (18,130 items). Loaded via
`manage.py import_catalog`. Embeddings generated via `manage.py generate_embeddings`.

### Genre Taxonomy

32 genres in Russian, matching the Okko catalog exactly:

Аниме, Артхаус, Биографии, Блоги, Боевики, Вестерны, Военное,
Детективы, Документальное, Драмы, Интервью, Историческое, Комедии,
Концерты, Короткий метр, Криминальное, Курсы, Мелодрамы, Музыкальное,
Мультфильмы, Презентации, Приключения, Природа, Путешествия, Семейное,
Советское, Триллеры, Ужасы, Фантастика, Фильмы для детей, Фитнес, Фэнтези

## ChatSession

Represents a single conversation session with a user. Stores the evolving
preference profile for session-based recommendation learning.

**Table**: `movies_chatsession`

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | BigAutoField | PK | Auto-generated primary key |
| `session_id` | UUIDField | unique, indexed | Frontend-generated session identifier |
| `session_token` | CharField(64) | indexed | Cryptographic token for session auth (`secrets.token_urlsafe(32)`) |
| `preference_vector` | VectorField(768) | nullable | EMA-updated preference embedding |
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
  "liked_genres": ["Комедии", "Приключения"],
  "disliked_genres": ["Ужасы"],
  "themes": ["космос", "путешествия"],
  "reference_films": ["Интерстеллар"]
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
