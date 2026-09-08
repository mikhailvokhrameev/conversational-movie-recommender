# Core ML Pipeline (`backend/core/`)

The `core/` package contains the recommendation engine's ML logic, independent
of Django's web framework layer. Every module is a pure Python function library
with no HTTP handling or database ORM calls (except `candidate_generation.py`
which queries the Movie model via Django ORM + pgvector).

## Architecture

```
User query (natural language)
  |
  v
ollama_client.aclassify_and_parse()   ──> category + structured intent (one JSON call)
  |                                        (includes mood detection)
  |                                        fallback: empty intent (semantic search still works)
  v
embedding_service.encode_query()      ──> 1024-dim query vector (BGE-M3)
  |
  v
candidate_generation.generate_candidates(query_vec, intent, query_text)
  |   1. HARD FILTERS (SQL WHERE, not scoring signals):
  |        - exclude movies matching negated genres
  |        - exclude/include movies by country
  |        - exclude movies below min_vote_average (nulls pass through)
  |        - exclude movies outside min/max_release_year (nulls pass through)
  |        - exclude movies outside min/max_runtime (nulls pass through)
  |        - exclude movies not matching original_languages
  |   2. TWO RETRIEVAL CHANNELS over the survivors:
  |        semantic -- exact cosine distance, top-100
  |        lexical  -- Postgres full-text over title/original_title/keywords, top-100
  |   3. Fuse the two rankings via RRF -> top-100 candidates
  v
scoring.score_candidates() over the whole candidate set
  |   min-max normalize each signal across the set, then:
  |   semantic:    cosine_sim(query_vec, movie_vec)   * 0.35
  |   metadata:    genre_overlap(intent, movie)       * 0.25
  |   session:     cosine_sim(session_vec, movie_vec) * 0.25
  |   popularity:  normalized TMDB popularity         * 0.15
  v
reranking.rerank_candidates(query, scored)   [optional, GPU cross-encoder]
  |   top-50 by score -> (query, movie) pairs -> blended 50/50 with scorer
  v
scoring.mmr_diversify(scored, top_n=5, lambda=0.7)
  |   Greedy MMR: balance relevance vs diversity
  v
Top-5 diverse recommendations
  |
  v
ollama_client.astream_explanation()  ──> streamed RAG explanation
  |
  v
session_manager.update_preference_vector()  ──> EMA blend into session vector
session_manager.track_explicit_preferences() ──> accumulate liked/disliked genres
```

## Modules

### `embedding_service.py` -- Vector Embeddings

Wraps `BAAI/bge-m3` (1024 dimensions).
The model is loaded lazily on first call and cached as a module-level singleton.

| Function | Input | Output | Use case |
|----------|-------|--------|----------|
| `get_model()` | -- | `SentenceTransformer` | Access the singleton model |
| `encode_texts(texts)` | `list[str]` | `np.ndarray (N, 1024)` | Batch embedding (catalog import) |
| `encode_query(query)` | `str` | `list[float]` (1024) | Single query at request time |
| `cosine_similarity(a, b)` | two vectors | `float [-1, 1]` | Similarity between any two vectors |

### `ollama_client.py` -- LLM Integration

Calls the Ollama container's HTTP API (`/api/chat`) for classification+intent
parsing and explanation generation. The request path uses the async variants
below; sync counterparts of the same functions exist for management commands
(`evaluate_scoring`, etc.) that run outside the ASGI event loop.

**Classification + intent parsing** (`aclassify_and_parse`): One combined JSON-mode
call replaces what used to be two separate classify/parse requests. Sends the
user query with a structured prompt requesting JSON output; Ollama's
`format: "json"` mode forces valid JSON, and the result is validated against
the `MessageIntent` Pydantic schema (one retry-with-repair on invalid category).
Extracts the routing category plus genres, mood, themes, negations, reference
films, country exclusions/inclusions, min vote average, min/max release year,
min/max runtime, and original languages. All of the latter are hard
constraints -- they get applied as SQL filters in `candidate_generation.py`,
not as scoring weights, so a movie violating one is excluded outright rather
than merely ranked lower. On failure (timeout, malformed JSON, Ollama down),
falls back to `_fallback_intent()`/`_fallback_message_intent()`, which returns
an empty intent. Semantic search via embeddings still works without parsed
intent -- it just loses metadata filtering.

**Explanation generation** (`astream_explanation`, `stream_explanation`): RAG pattern.
Movie metadata and descriptions are injected as context, and the LLM writes 1-2
sentences per movie explaining the match. The streaming variant yields tokens
for progressive frontend display.

| Function | Ollama API | Timeout | Fallback |
|----------|-----------|---------|----------|
| `aclassify_and_parse(message)` | `POST /api/chat` (JSON mode) | 45s | `_fallback_message_intent()` |
| `astream_explanation(query, movies)` | `POST /api/chat` (stream) | 120s | Silent stop |
| `astream_conversational(message, context)` | `POST /api/chat` (stream) | 120s | Silent stop |
| `ais_available()` | `GET /` | 5s | Returns `False` |

### `candidate_generation.py` -- Hard Filters + Hybrid Retrieval

Applies hard filters, then retrieves the top-N semantically similar movies
via exact pgvector cosine search (no ANN index -- see ML.md) before passing
candidates to the reranker.

1. **Hard filters** (all SQL `WHERE`/`exclude`, not scoring signals):
   - `exclude(genres__contains=negated_genre)` for each negated genre
   - `exclude(country__contains=excluded_country)` for each excluded country
   - `filter(country__contains=included_country)` (OR-ed) for country inclusions
   - `filter(vote_average__gte=min_vote_average)` (movies with no rating pass through)
   - `filter(release_date__year__gte/__lte=min/max_release_year)` (movies with no date pass through)
   - `filter(runtime__gte/__lte=min/max_runtime)` (movies with no runtime pass through)
   - `filter(original_language__in=original_languages)` (exact match, no null pass-through --
     original_language is always known for an imported row)
2. **Semantic channel**: `CosineDistance` ordering on the embedding column,
   limit=100. No ANN index -- see ML.md for the measurements behind that choice.
3. **Lexical channel**: full-text match against `search_vector` (title/original_title
   weight A, keywords weight B -- the catalog has no cast/crew data to index),
   ranked by `ts_rank`, limit=100. Query terms are OR-ed, since a conversational
   sentence would match nothing under AND. Skipped entirely when the message
   yields no usable terms.
4. **RRF fusion**: `weight / (60 + rank)` summed per movie across channels.
   Rank-based because a cosine distance and a `ts_rank` are not comparable
   quantities. Semantic is weighted 1.0 and lexical 0.7, so lexical only loses
   ties. Fusion decides pool membership; `scoring.py` re-ranks the pool.

Returns a list of movie dicts with all metadata + embedding for downstream scoring.

### `scoring.py` -- Hybrid Reranking + MMR Diversification

**Scoring** -- four signals, weighted sum:

- **Semantic (0.35)**: Cosine similarity between the query embedding and the movie's
  pre-computed embedding. Mapped from [-1, 1] to [0, 1] via `(sim + 1) / 2`.
- **Metadata (0.25)**: Genre overlap ratio between LLM-extracted intent genres and
  movie genres. Direct matching only, no indirect mood-to-genre lookup.
- **Session (0.25)**: Cosine similarity between the session preference vector and the
  movie embedding. Zero on the first turn (no session vector yet).
- **Popularity (0.15)**: TMDB's popularity metric. A scoring signal rather than a
  hard filter, deliberately -- popularity is unbounded and non-linear, so there's
  no absolute threshold an LLM could reliably pick (unlike vote_average's bounded
  0-10 scale, which *is* a hard filter above).

Signals are min-max normalized across the candidate set before the weighted
sum, so the declared weights actually hold. Raw signals have very different
spreads (semantic cosines cluster in a narrow band; genre overlap spans the
full 0-1 range), and a signal's real influence is `weight * spread`. Without
normalization metadata's 0.25 could outrank semantic's 0.35. A signal identical
across every candidate carries no ranking information and collapses to a
neutral 0.5. Pre-normalization values are kept per candidate under
`raw_scores` for debugging and evaluation.

Scoring is set-level (`score_candidates`) rather than per-movie, because
normalization needs the full candidate pool to know each signal's range.

Weights live in `params.yaml` under `scoring.weights`. Must sum to 1.0.
Evaluated via `manage.py evaluate_scoring` and documented in `docs/ML.md`.

**MMR diversification** (Carbonell & Goldstein, 1998):
After scoring, `mmr_diversify()` selects the top-N results greedily. The first
pick is the highest-scored candidate. Each subsequent pick maximizes:
`lambda * relevance_score - (1 - lambda) * max_similarity_to_already_selected`.
Default lambda=0.7 (relevance-biased). This prevents returning 5 movies by
the same director or in the same sub-genre.

### `reranking.py` -- Cross-Encoder Reranking

The scorer compares a query *embedding* to a movie *embedding* -- two vectors
produced independently that never see each other. A cross-encoder reads the
query and the movie text together in a single forward pass and scores their
relevance directly, which is what lets it catch relationships a bi-encoder
structurally cannot. The cost is that nothing can be precomputed: every
(query, movie) pair is a model call.

So it runs on a short slice. The top `reranking.top_k` (default 50) candidates by
scorer total are paired with the query, scored, and blended:

    total = reranking.weight * rerank_norm + (1 - reranking.weight) * scorer_norm

Both sides are min-max normalized across the slice first, for the same reason
the scorer normalizes its own signals -- otherwise the weight would not mean
what it says. Blending rather than replacing keeps the session and genre
signal, which the cross-encoder knows nothing about.

The call returns only the reranked slice. Candidates below the cut ranked
below 50 items and cannot reach a top-5 result, and keeping them would mix
two incompatible `total` scales in one list.

**Model and device are one decision.** `bge-reranker-v2-m3` (the default) is
an XLM-RoBERTa-large backbone: ~302M body parameters. Measured on this GPU
(GTX 1080 Ti): 20 pairs ~0.26s, 50 pairs ~0.64s, 100 pairs ~1.22s -- versus
~5.6s for 20 pairs on CPU. `reranking.device: auto` picks cuda when present,
which is what makes this model viable; without a GPU, switch
`reranking.model` to a small cross-encoder or set `reranking.enabled: false`
rather than just moving this one to CPU.

**Fail-safe**: a model that will not load or a prediction that raises logs and
returns the candidates in scorer order. Reranking is a quality improvement,
never a correctness dependency, so it cannot break a request. Disable
entirely with `reranking.enabled: false` in params.yaml.

### `session_manager.py` -- Preference Learning

Tracks per-session preferences across conversation turns using two mechanisms:

**Preference vector (implicit)**: Exponential moving average of query embeddings.
`new = alpha * query_vec + (1 - alpha) * current_vec`, then L2-normalized.
With `alpha=0.7`, the most recent query contributes 70% of the signal. Turns
classified `refinement` use `alpha_refinement=0.92` instead, so an explicit
correction dominates the vector rather than blending with the turn before it
(see Design Decisions).

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
Test set: `data/test_queries.json` (18 hand-curated English queries with relevance judgments).
This is a genre-overlap heuristic, not the manually labeled golden-set eval
tracked in `TODOS.md` -- see there before treating either as ground truth.

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

**Why refinement turns use a higher alpha**: Negated genres are already excluded
via hard filters in `candidate_generation.py`, but the EMA vector still feeds the
`session` scoring signal over whatever survives those filters. At the default
alpha=0.7, 30% of a strongly-aligned prior-turn direction could still bias that
signal back toward what the user just moved away from ("не ужасы, а повеселее"
still nudging ranking toward horror-adjacent moods). Raising alpha to 0.92 on
`refinement` turns makes the new query dominate the vector instead of blending,
without discarding session history entirely (see `session.alpha_refinement` in
params.yaml). The candidate pool itself is unaffected -- `generate_candidates`
builds it fresh each turn from the current query embedding and current-turn
hard filters, not from session state, so this only changes ranking, not retrieval.

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
