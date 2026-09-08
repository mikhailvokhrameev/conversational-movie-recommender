# ML Design Decisions and Evaluation

This document explains the reasoning behind each ML decision in the
recommendation pipeline. It covers what was chosen, what was considered,
and what the evaluation results show.

## System Architecture

This is a retrieval-augmented recommendation system, not standard RAG.
The distinction matters:

- **Standard RAG**: retrieve documents, stuff into LLM, LLM generates answer.
  The LLM IS the output and decides what's relevant.
- **This system**: retrieve candidates via ANN search, score with a
  deterministic algorithm, ask LLM to explain results. The algorithm
  decides what's relevant. The LLM serves two helper roles: intent
  parsing (input) and explanation generation (output).

This makes the recommendation logic explainable, testable, and tunable,
which pure RAG is not.

## Embedding Model

**Choice**: `BAAI/bge-m3` (1024 dimensions)

**Why**: The catalog is TMDB, English-language. BGE-M3 is a strong general-purpose
multilingual embedder (still useful headroom if non-English queries come in),
produces 1024-dim vectors, and runs on both GPU and CPU. No fine-tuning needed.

**Migration note**: This replaced `paraphrase-multilingual-mpnet-base-v2`
(768-dim) in the same change that swapped the catalog from the old
Russian-language source to TMDB. The swap and the embedder upgrade shipped
together -- the full catalog re-embed was happening anyway, so there is no
valid before/after recall@k comparison between the two embedders (catalog and
model changed at once). See `TODOS.md`'s golden-set item for measuring BGE-M3
on its own merits once it lands.

**Embedding text**: Each movie is embedded as
`"{serial_name}. {genres joined}. {description[:500]}. Keywords: {keywords[:10] joined}"`.
This captures title, genre signal, semantic content, and curated theme tags in one vector.

## LLM (Ollama)

**Choice**: Ollama running `qwen3:8b` (Q4_K_M, pulled from Ollama's own model
library), set by `llm.model` in `params.yaml`

**Why**: Runs locally (no API costs). Plain Qwen3 8B rather than a
Russian-adapted variant, because the product moved to English along with the
TMDB catalog swap -- there's no longer a reason to pay for a Russian-specialized
tokenizer. ~5.0GB at Q4_K_M, which fits the 11GB card alongside the reranker
and embedder (~10.4-10.9GB total; see `params.yaml`'s VRAM budget comment --
noticeably tighter than the old mpnet embedder's footprint, since BGE-M3 is
in the same XLM-RoBERTa-large size class as the reranker).

**Reasoning mode is disabled** (`llm.thinking: false`). This is a hybrid
reasoning model: left alone it emits a `<think>...</think>` span before every
answer. That breaks JSON-mode intent parsing and adds seconds to each of the
two sequential LLM calls per turn (classify+parse, then explanation). The
flag is sent to Ollama on every request, and `ollama_client` additionally
strips reasoning spans from both parsed JSON and the streamed output, so an
Ollama build that ignores the flag degrades to slower rather than broken.

**Two roles**:
1. **Classification + intent parsing**: one combined call (`aclassify_and_parse`)
   returns the message category plus genres, mood, themes, negations, reference
   films, country exclusions/inclusions, min vote average, min/max release year,
   min/max runtime, and original languages, validated against a Pydantic schema
   with a repair retry on invalid JSON. Uses JSON mode for structured output.
2. **Explanation generation**: writes explanations of why each recommended
   movie matches the query (RAG pattern).

**Genre normalization**: The LLM often returns genre names in the wrong form
(e.g., "sci-fi" instead of "Science Fiction"). An embedding-based normalizer
maps LLM output to exact catalog genre names (TMDB's 19-genre taxonomy) by
cosine similarity. Threshold is `intent.genre_match_threshold` in `params.yaml`
(default 0.5).

**Fallback**: When Ollama is unavailable, `aclassify_and_parse` falls back to
an empty intent. Semantic search via embeddings still works without parsed
intent -- it just loses metadata filtering. This is preferable to broken
keyword parsing that silently produces wrong results.

## Candidate Generation

**Choice**: exact pgvector cosine search, no ANN index, top-100 candidates

**Why no index at all**: The original question was framed as "HNSW or
IVFFlat", which skipped the prior question of whether approximate search is
warranted. At ~50K items it is not. A brute-force top-100 over 50K x 1024
float32 vectors measures a few milliseconds in numpy; in Postgres, with row
overhead, a sequential scan lands in the tens of milliseconds. The same
request spends roughly 2 seconds in the Ollama intent call, so the index was
optimising under 2% of request latency in exchange for giving up exact recall.

**Why it actively hurt**: hard constraints (negations, country
exclusions/inclusions, rating floor, release year range, runtime range,
original language) are applied as WHERE clauses before the vector ordering.
An HNSW scan walks its graph and post-filters, so selective filters can leave
far fewer than the requested 100 candidates. Exact search applies the filters
first and then ranks whatever genuinely qualifies.

**When to revisit**: if the catalog grows by roughly an order of magnitude
(~500K+ items), measure again and reintroduce an index if the scan shows up
in the latency budget.

**Negation as hard filter**: User negations ("not horror") are applied
as hard filters in candidate generation, not as scoring signals. A user
saying "not horror" means zero tolerance. Making it a scoring signal
(metadata=0.0) still allows horror movies through via high semantic similarity.
Hard filtering eliminates the leakage entirely.

## Lexical Retrieval Channel

**Choice**: Postgres `tsvector` over title/original_title/keywords, fused with
the semantic channel by Reciprocal Rank Fusion.

**Why a second channel**: embeddings encode meaning, not identity. A query
naming a specific film ("something like Interstellar") can miss the exact
record entirely, because nothing in the vector space privileges the literal
name. Full-text search matches names and understands nothing, which is the
complementary failure mode. Neither channel subsumes the other.

**What is indexed**: `serial_name` and `original_title` at weight A,
`keywords` (curated franchise/character/theme tags) at weight B -- the
catalog has no director/cast data to index. `description` is deliberately
excluded -- it is long free text that would dominate the index by token count
and match on ordinary vocabulary, and it is exactly what the semantic channel
already handles. Keeping it out preserves the split: lexical finds names,
semantic finds meaning.

**Text search config**: `english` (snowball stemmer + stopwords), matching
the TMDB catalog's English titles and keywords.

**OR, not AND**: query terms are OR-ed. The input is a conversational
sentence, so requiring every term to match would return nothing; `ts_rank`
then sorts by how much of the query landed and in which weight class.

**Why RRF instead of a weighted score**: a cosine distance and a `ts_rank`
are not comparable quantities, and normalizing them against each other would
be inventing a relationship that does not exist. RRF fuses by *rank*
(`weight / (60 + rank)`), which is scale-free by construction. Note the
contrast with hybrid scoring below, which uses min-max normalization instead:
there, all three signals rank the same pool and their magnitudes are
meaningful, so discarding magnitude would lose real information.

**Channel weights**: semantic 1.0, lexical 0.7. Semantic carries signal for
every query; lexical only when the user names something. The weighting means
lexical loses ties rather than being suppressed.

**Interaction with reranking**: fusion decides which candidates enter the
pool, not the final order -- the three-signal scorer has no lexical term, so
a movie retrieved purely on an exact title match would enter the pool and
then rank low. The cross-encoder reranker closes that gap: it reads the query
and the movie text together, so a literal title match scores highly on
relevance regardless of what the embeddings thought.

## Hybrid Scoring

Four signals combined via weighted sum, each min-max normalized across the
candidate set first:

| Signal | Default Weight | What it measures |
|--------|---------------|------------------|
| Semantic | 0.35 | Cosine similarity between query and movie embeddings |
| Metadata | 0.25 | Genre overlap between LLM-extracted intent and movie genres |
| Session | 0.25 | Cosine similarity between session preference vector and movie |
| Popularity | 0.15 | TMDB's popularity metric (a scoring signal, not a hard filter -- it's unbounded and non-linear, unlike vote_average's bounded 0-10 scale) |

**Cosine similarity mapping**: Raw cosine similarity ranges [-1, 1].
Mapped to [0, 1] via `(sim + 1) / 2` so negative similarity contributes 0,
not negative weight.

**Why normalize before summing**: the raw signals have very different spreads.
Semantic and session cosines cluster in a narrow band, popularity is unbounded
and heavily right-skewed, while genre overlap spans the full 0-1 range and
moves in large discrete jumps. A signal's real influence on the ranking is
`weight * spread`, not weight alone, so un-normalized the declared weights
would not describe the actual behaviour. Min-max normalizing each signal
across the candidate pool makes every signal span the same range, so
influence equals the weight as written.

A signal identical across every candidate carries no ranking information and
collapses to a neutral 0.5 rather than being stretched across the full range
by numerical noise. Pre-normalization values are retained per candidate under
`raw_scores` for debugging and evaluation.

This is deliberately a different technique from the RRF used to fuse the two
retrieval channels above. Retrieval fuses incomparable scores across
independent systems, where only rank is trustworthy. Scoring ranks one shared
pool with three commensurable signals, where magnitude is real information
worth keeping.

**Weight tuning**: Pre-swap grid search (old Okko catalog, old embedder,
3-signal weight scheme) showed minimal impact on LLM-judged relevance across
configurations, suggesting semantic similarity dominated regardless of weight
allocation. That result no longer applies -- the catalog, embedder, and
weight scheme (a `popularity` signal was added) have all changed since. Needs
re-running against the current pipeline; see the golden-set item in
`TODOS.md`.

**Configurable**: weights live in `params.yaml` under `scoring.weights` and
must sum to 1.0. That file is the only place they can be changed.

## Cross-Encoder Reranking

**Choice**: `BAAI/bge-reranker-v2-m3` on the GPU, applied to the top 50 scored
candidates, blended 50/50 with the scorer total.

**Why rerank at all**: the scorer is a bi-encoder comparison -- query vector
against movie vector, each computed without ever seeing the other. A
cross-encoder reads both together in one forward pass, which is a strictly
richer view and typically improves ordering more than any weight tuning on
the existing signals. It is also what makes the lexical channel pay off:
an exact title match scores highly on direct relevance even when the
embeddings disagreed.

**Model and device are one decision**: bge-reranker-v2-m3 is an
XLM-RoBERTa-large backbone at ~302M body parameters. Measured on this GPU
(GTX 1080 Ti): 20 pairs ~0.26s, 50 pairs ~0.64s, 100 pairs ~1.22s -- versus
~5.6s for 20 pairs on CPU. The same model is either comfortably interactive
or completely unusable depending on where it runs, so `reranking.device: auto`
(cuda when present) is not an optimisation here, it is what makes the choice
viable at all.

On a machine without a usable GPU, do not simply move this model to CPU.
Either switch `reranking.model` to a small multilingual cross-encoder such as
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (~21M body parameters, ~14x less
compute), or set `reranking.enabled: false`.

**VRAM**: ~2.2GB at fp32 (sentence-transformers does not downcast this model
to fp16 automatically -- budget for the larger figure), on top of ~5.0GB for
the LLM and ~2.2GB for the BGE-M3 embedder -- roughly 10.4-10.9GB of the 11GB
card including CUDA context and activations. Noticeably tighter than the old
mpnet embedder's footprint; see `params.yaml`'s VRAM budget comment for the
current breakdown.

**Why blend rather than replace**: the cross-encoder judges query-document
relevance and knows nothing about session preference or extracted genre
intent. Replacing the scorer total would discard personalization entirely.
Both sides are min-max normalized across the slice before blending, so
`reranking.weight` means what it says.

**Why 50**: every candidate is a model forward pass, so `reranking.top_k` is
the primary latency dial. Raised from 20 to 50 once this ran on GPU: +0.38s
for a 2.5x larger reranked pool, against a ~2s LLM call already in the same
request.

**Fail-safe**: a model that fails to load, or a prediction that raises, logs
and returns candidates in scorer order. A failed download must not break
recommendations, and the failure is cached so it is not retried per request.

**Not yet measured**: whether reranking improves results here is unverified.
It is a well-established technique and the reasoning is sound, but the
golden-set evaluation is the thing that would actually prove it. Treat
`reranking.weight` and `reranking.top_k` as untuned defaults until then.

## MMR Diversification

**Choice**: Maximal Marginal Relevance (Carbonell & Goldstein, 1998)

**Why**: Without diversification, top-5 results for "I want a thriller" might
return 5 movies from the same franchise or the same country. MMR ensures
variety by penalizing each subsequent pick for similarity to already-selected
results.

**Formula**: `MMR = lambda * relevance - (1 - lambda) * max_similarity_to_selected`

**lambda = 0.7** (relevance-biased): Higher values prioritize relevance over
diversity. 0.7 means 70% relevance, 30% diversity penalty.

**Why MMR over genre-based post-filtering**: Genre rules ("max 2 per genre")
only handle one dimension. MMR uses embedding similarity to catch all forms
of redundancy: same director, same sub-genre, same narrative structure.

## Session-Based Preference Learning

**Choice**: Exponential Moving Average (EMA) of query embeddings

**Formula**: `new_vec = alpha * query_vec + (1 - alpha) * current_vec`,
then L2-normalized.

**alpha = 0.7** (`session.alpha` in params.yaml): Recent queries contribute
70% of the signal. After 5 turns, the first query's weight decays to ~0.8%.

**alpha_refinement = 0.92** (`session.alpha_refinement`): used instead of
`alpha` on turns classified `refinement` ("something funnier?", "no horror").
Hard
filters (negations, exclusions) already drop disallowed genres from the
candidate pool outright, but the EMA vector still feeds the `session`
scoring signal on whatever survives those filters -- at alpha=0.7, 30% of a
strongly-aligned prior-turn direction could still bias that signal back
toward what the user just corrected away from. The higher alpha makes an
explicit refinement dominate the vector instead of blending.

**Why EMA over simple averaging**: Simple averaging gives equal weight to all
turns. An early exploratory query permanently dilutes the signal from a later
specific query. EMA makes recent queries dominate.

**Explicit preferences**: Alongside the implicit vector, the session tracks
liked genres, disliked genres, themes, and reference films extracted from
parsed intent. Not currently read back into scoring or filtering -- exposed
read-only via `GET /api/sessions/<id>/`.

## Evaluation Framework

### Metrics

| Metric | What it measures | Implementation |
|--------|-----------------|----------------|
| Precision@k | Fraction of top-k results matching relevant genres | `evaluation.precision_at_k()` |
| NDCG@k | Position-aware relevance (supports graded scores) | `evaluation.ndcg_at_k()` |
| Diversity | 1 - mean pairwise cosine similarity among results | `evaluation.diversity()` |
| Novelty | How non-obvious the recommendations are (genre rarity) | `evaluation.novelty()` |
| Coverage | Fraction of catalog appearing in any recommendation | `evaluation.coverage()` |
| LLM Relevance | Ollama rates each recommendation 1-5 for relevance | `evaluation.llm_relevance_score()` |
| Negation Violations | Count of results matching negated genres | `evaluate_scoring._score_query()` |

### LLM-as-Judge

When `--llm-judge` is enabled, the same Ollama model rates each recommended
movie's relevance to the query on a 1-5 scale. These graded scores feed into
NDCG for position-aware quality measurement.

**Known limitation (circularity)**: The same model that parses intent also
judges relevance. This creates evaluation bias. A production system would
use a stronger or different model for judging. For a portfolio project,
this is documented honestly as a limitation.

### Running Evaluation

```bash
# Genre-based evaluation (fast)
docker compose exec backend python manage.py evaluate_scoring

# With LLM-as-judge graded relevance (slow, ~2s per movie)
docker compose exec backend python manage.py evaluate_scoring --llm-judge

# Grid search over weight space (fast, genre-based)
docker compose exec backend python manage.py evaluate_scoring --sweep

# Grid search with LLM judge (6 configs, ~12 min on GPU)
docker compose exec backend python manage.py evaluate_scoring --sweep --llm-judge

# Custom weights
docker compose exec backend python manage.py evaluate_scoring --weights 0.6,0.2,0.2
```

### Evaluation Results

**Stale -- pre-swap, do not treat as current.** The numbers that used to live
here were measured against the old Okko catalog (18,130 items), the old
mpnet embedder (768-dim), the old 3-signal weight scheme, and a Russian-language
test set. All four of those changed in the TMDB catalog swap (`28f0194`), so
none of the old figures describe the current system. No fresh numbers have
replaced them yet -- `data/test_queries.json` was updated to the new English
genre taxonomy, but re-running `evaluate_scoring` and reporting real results
here is still open work. The bigger, non-circular replacement for this whole
section is the manually-labeled golden-set eval tracked in `TODOS.md`;
prefer that over re-populating this table with more LLM-judge numbers, per
the circularity caveat below.

## Production Extensions (not implemented)

Documented here for completeness. A production system would add:

- **Time-of-day context signals**: Hypothesis that evening users prefer longer
  films, weekend users prefer family content. Requires A/B testing to validate.
- **True collaborative filtering**: Requires user accounts, watch history, and
  rating data. Would use user-item matrix (ALS/BPR) blended with content-based scores.
- **Model evaluation pipeline**: Automated NDCG/MAP/MRR measurement with CI
  integration, regression detection on model changes.
- **A/B testing framework**: For comparing scoring weight configurations with
  real user engagement metrics (click-through, watch completion).
- **Cross-model evaluation**: Using a stronger LLM (14B+) or different model
  family for judging relevance, eliminating the circularity limitation.
