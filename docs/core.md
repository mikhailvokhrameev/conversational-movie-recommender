# Core ML Pipeline (`backend/core/`)

The `core/` package contains the recommendation engine's ML logic, independent
of Django's web framework layer. Every module is a pure Python function library
with no HTTP handling or database ORM calls (except `candidate_generation.py`
which queries the Movie model via Django ORM + pgvector).

## Architecture

```
User query (Russian natural language)
  |
  ├──[PARALLEL]──> ollama_client.parse_intent()     ──> structured intent (JSON)
  │                  (includes mood detection)
  │                  fallback: empty intent (semantic search still works)
  │
  └──[PARALLEL]──> embedding_service.encode_query()  ──> 768-dim query vector
  |
  v
candidate_generation.generate_candidates(query_vec, intent)
  |   1. HARD FILTERS (SQL WHERE, not scoring signals):
  |        - exclude movies matching negated genres
  |        - exclude movies matching excluded countries
  |        - exclude movies above max_age_rating (nulls pass through)
  |        - exclude movies older than min_release_year (nulls pass through)
  |   2. Optional: filter by content_type from intent
  |   3. Order survivors by exact cosine distance, take top-100
  v
scoring.score_candidates() over the whole candidate set
  |   min-max normalize each signal across the set, then:
  |   semantic:  cosine_sim(query_vec, movie_vec)  * 0.4
  |   metadata:  genre_overlap(intent, movie)      * 0.3
  |   session:   cosine_sim(session_vec, movie_vec) * 0.3
  v
scoring.mmr_diversify(scored, top_n=5, lambda=0.7)
  |   Greedy MMR: balance relevance vs diversity
  v
Top-5 diverse recommendations
  |
  v
ollama_client.generate_explanation()  ──> Russian-language RAG explanation
  |
  v
session_manager.update_preference_vector()  ──> EMA blend into session vector
session_manager.track_explicit_preferences() ──> accumulate liked/disliked genres
```

## Modules

### `embedding_service.py` -- Vector Embeddings

Wraps `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (768 dimensions).
The model is loaded lazily on first call and cached as a module-level singleton.

| Function | Input | Output | Use case |
|----------|-------|--------|----------|
| `get_model()` | -- | `SentenceTransformer` | Access the singleton model |
| `encode_texts(texts)` | `list[str]` | `np.ndarray (N, 768)` | Batch embedding (catalog import) |
| `encode_query(query)` | `str` | `list[float]` (768) | Single query at request time |
| `cosine_similarity(a, b)` | two vectors | `float [-1, 1]` | Similarity between any two vectors |

### `ollama_client.py` -- LLM Integration

Calls the Ollama container's HTTP API (`/api/chat`) for three tasks:

**Intent parsing** (`parse_intent`): Sends the user query with a structured prompt
requesting JSON output. Ollama's `format: "json"` mode forces valid JSON.
Extracts genres, mood, themes, negations, reference films, country exclusions,
max age rating, and min release year. The latter three are hard constraints
(country_exclusions, max_age_rating, min_release_year) -- they get applied as
SQL filters in `candidate_generation.py`, not as scoring weights, so a movie
violating one is excluded outright rather than merely ranked lower.
On failure (timeout, malformed JSON, Ollama down), falls back to `_fallback_intent()`
which returns an empty intent. Semantic search via embeddings still works without
parsed intent -- it just loses metadata filtering.

**Explanation generation** (`generate_explanation`, `stream_explanation`): RAG pattern.
Movie metadata and descriptions are injected as context, and the LLM writes 1-2
sentences per movie explaining the match. The streaming variant yields tokens
for progressive frontend display.

| Function | Ollama API | Timeout | Fallback |
|----------|-----------|---------|----------|
| `parse_intent(query)` | `POST /api/chat` (JSON mode) | 60s | Keyword extraction |
| `generate_explanation(query, movies)` | `POST /api/chat` | 120s | Empty string |
| `stream_explanation(query, movies)` | `POST /api/chat` (stream) | 120s | Silent stop |
| `is_available()` | `GET /` | 5s | Returns `False` |

### `candidate_generation.py` -- Hard Filters + Exact Vector Search

Applies hard filters, then retrieves the top-N semantically similar movies
via exact pgvector cosine search (no ANN index -- see ML.md) before passing
candidates to the reranker.

1. **Hard filters** (all SQL `WHERE`/`exclude`, not scoring signals):
   - `exclude(genres__contains=negated_genre)` for each negated genre
   - `exclude(country__contains=excluded_country)` for each excluded country
   - `filter(age_rating__lte=max_age_rating)` (movies with no rating pass through)
   - `filter(release_date__year__gte=min_release_year)` (movies with no date pass through)
2. **Optional filter**: content_type from intent
3. **Exact vector search**: `CosineDistance` ordering on the embedding column,
   limit=100. No ANN index -- see ML.md for the measurements behind that choice.

Returns a list of movie dicts with all metadata + embedding for downstream scoring.

### `scoring.py` -- Hybrid Reranking + MMR Diversification

**Scoring** -- three signals, weighted sum:

- **Semantic (0.4)**: Cosine similarity between the query embedding and the movie's
  pre-computed embedding. Mapped from [-1, 1] to [0, 1] via `(sim + 1) / 2`.
- **Metadata (0.3)**: Genre overlap ratio between LLM-extracted intent genres and
  movie genres. Direct matching only, no indirect mood-to-genre lookup.
- **Session (0.3)**: Cosine similarity between the session preference vector and the
  movie embedding. Zero on the first turn (no session vector yet).

Signals are min-max normalized across the candidate set before the weighted
sum, so the declared weights actually hold. Raw signals have very different
spreads (semantic cosines cluster in a narrow band; genre overlap spans the
full 0-1 range), and a signal's real influence is `weight * spread`. Without
normalization metadata's 0.3 outranked semantic's 0.4. A signal identical
across every candidate carries no ranking information and collapses to a
neutral 0.5. Pre-normalization values are kept per candidate under
`raw_scores` for debugging and evaluation.

Scoring is set-level (`score_candidates`) rather than per-movie, because
normalization needs the full candidate pool to know each signal's range.

Weights are configurable via `SCORE_WEIGHT_*` environment variables. Must sum to 1.0.
Evaluated via `manage.py evaluate_scoring` and documented in `docs/ML.md`.

**MMR diversification** (Carbonell & Goldstein, 1998):
After scoring, `mmr_diversify()` selects the top-N results greedily. The first
pick is the highest-scored candidate. Each subsequent pick maximizes:
`lambda * relevance_score - (1 - lambda) * max_similarity_to_already_selected`.
Default lambda=0.7 (relevance-biased). This prevents returning 5 movies by
the same director or in the same sub-genre.

### `session_manager.py` -- Preference Learning

Tracks per-session preferences across conversation turns using two mechanisms:

**Preference vector (implicit)**: Exponential moving average of query embeddings.
`new = alpha * query_vec + (1 - alpha) * current_vec`, then L2-normalized.
With `alpha=0.7`, the most recent query contributes 70% of the signal.

**Explicit preferences**: Accumulates liked genres, disliked genres, themes, and
reference films extracted from parsed intent.

### `evaluation.py` -- Offline Quality Metrics

Reusable evaluation framework for measuring recommendation quality:

| Metric | What it measures |
|--------|-----------------|
| `precision_at_k` | Fraction of top-k results that are relevant |
| `recall_at_k` | Fraction of relevant items that appear in top-k |
| `ndcg_at_k` | Position-aware relevance (higher = relevant items ranked higher) |
| `coverage` | Fraction of catalog that appears in any recommendation |
| `diversity` | 1 - mean pairwise cosine similarity (higher = more diverse results) |

Used by `manage.py evaluate_scoring` and the ablation notebook.
Test set: `data/test_queries.json` (20 hand-curated Russian queries with relevance judgments).

## Design Decisions

**Why negation is a hard filter, not a scoring signal**: A user saying "not horror"
means zero tolerance for horror results. Making it a scoring signal (metadata=0.0)
still allows horror movies through via high semantic similarity. Hard filtering in
candidate generation eliminates the leakage entirely.

**Why MMR over genre-based post-filtering**: Genre-based rules ("max 2 per genre")
only handle one dimension of diversity. MMR uses embedding similarity to catch
all forms of redundancy: same director, same sub-genre, same narrative structure.
One algorithm, multi-dimensional diversity.

**Why cosine similarity is mapped to [0, 1]**: Raw cosine similarity ranges from -1
to 1. For scoring purposes, negative similarity should contribute 0, not negative
weight. The mapping `(sim + 1) / 2` gives: -1 to 0, 0 to 0.5, 1 to 1.0.

**Why EMA over simple averaging**: Simple averaging gives equal weight to all turns.
An early exploratory query permanently dilutes the signal from a later specific
query. EMA with alpha=0.7 makes recent queries dominate.

**Why the fallback returns empty intent instead of keyword parsing**: An earlier
version used Russian keyword stem matching as a fallback. It had two bugs:
(1) no negation scoping -- "хочу триллер, без ужасов" put both genres in the
negation list; (2) substring matching missed context ("не веселое" detected as
happy). Empty intent is safer: semantic search via embeddings still produces
relevant results, the user just loses genre filtering temporarily.

**Why mood detection is unified into Ollama**: The previous standalone
`mood_detector.py` used substring matching on Russian word stems. It missed
context, negation, and implicit mood. Ollama already parses the full query
and returns a mood field -- the separate module was doing the same job worse.

## Related

- [API Reference](api.md) for HTTP endpoints that call the core pipeline
- [Data Models](models.md) for Movie and ChatSession schemas
- [ML Decisions](ML.md) for evaluation methodology and parameter tuning
