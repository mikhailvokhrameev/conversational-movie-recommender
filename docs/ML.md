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

**Choice**: `paraphrase-multilingual-mpnet-base-v2` (768 dimensions)

**Why**: The catalog is Russian-language. This model handles Russian text
well (trained on parallel multilingual corpora), produces 768-dim vectors
(standard), and runs on both GPU and CPU. No fine-tuning needed.

**Alternative considered**: Fine-tuned Russian BERT models. Rejected because
the catalog descriptions are general-purpose text, not domain-specific jargon.
The multilingual model captures semantic similarity well enough for movie
descriptions.

**Embedding text**: Each movie is embedded as
`"{serial_name}. {genres joined}. {description[:500]}"`.
This captures title, genre signal, and semantic content in one vector.

## LLM (Ollama)

**Choice**: Ollama running `RefalMachine/RuadaptQwen3-8B-Hybrid-GGUF` (Q4_K_M),
set by `llm.model` in `params.yaml`

**Why**: Runs locally (no API costs) and is explicitly adapted for Russian,
which matters for a Russian-language catalog where the generic Qwen tokenizer
spends more tokens per word. ~5GB at Q4_K_M, which fits the 11GB card
alongside the reranker and embedder (~8.2GB total).

**Reasoning mode is disabled** (`llm.thinking: false`). This is a hybrid
reasoning model: left alone it emits a `<think>...</think>` span before every
answer. That breaks JSON-mode intent parsing and adds seconds to each of the
two sequential LLM calls per turn (classify+parse, then explanation). The
flag is sent to Ollama on every request, and `ollama_client` additionally
strips reasoning spans from both parsed JSON and the streamed output, so an
Ollama build that ignores the flag degrades to slower rather than broken.

**Two roles**:
1. **Classification + intent parsing**: one combined call (`aclassify_and_parse`)
   returns the message category plus genres, mood, themes, negations, and
   reference films, validated against a Pydantic schema with a repair retry
   on invalid JSON. Uses JSON mode for structured output.
2. **Explanation generation**: writes Russian-language explanations of why
   each recommended movie matches the query (RAG pattern).

**Genre normalization**: The LLM often returns genre names in wrong form
(e.g., "комедия" instead of "Комедии"). An embedding-based normalizer
maps LLM output to exact catalog genre names by cosine similarity.
Threshold is `intent.genre_match_threshold` in `params.yaml` (default 0.5).

**Fallback**: When Ollama is unavailable, `parse_intent` returns empty intent.
Semantic search via embeddings still works without parsed intent -- it just
loses metadata filtering. This is preferable to broken keyword parsing
that silently produces wrong results.

## Candidate Generation

**Choice**: exact pgvector cosine search, no ANN index, top-100 candidates

**Why no index at all**: The original question was framed as "HNSW or
IVFFlat", which skipped the prior question of whether approximate search is
warranted. At 18,130 items it is not. A brute-force top-100 over 18,130 x 768
float32 vectors measures ~1.2 ms in numpy; in Postgres, with row overhead, a
sequential scan lands in the tens of milliseconds. The same request spends
roughly 2 seconds in the Ollama intent call, so the index was optimising
under 2% of request latency in exchange for giving up exact recall.

**Why it actively hurt**: hard constraints (negations, country exclusions,
age rating, release year) are applied as WHERE clauses before the vector
ordering. An HNSW scan walks its graph and post-filters, so selective
filters can leave far fewer than the requested 100 candidates. Exact search
applies the filters first and then ranks whatever genuinely qualifies.

**When to revisit**: if the catalog grows by roughly an order of magnitude
(~200K+ items), measure again and reintroduce an index if the scan shows up
in the latency budget.

**Negation as hard filter**: User negations ("не хочу ужасы") are applied
as hard filters in candidate generation, not as scoring signals. A user
saying "not horror" means zero tolerance. Making it a scoring signal
(metadata=0.0) still allows horror movies through via high semantic similarity.
Hard filtering eliminates the leakage entirely.

## Lexical Retrieval Channel

**Choice**: Postgres `tsvector` over title/director/actors, fused with the
semantic channel by Reciprocal Rank Fusion.

**Why a second channel**: embeddings encode meaning, not identity. A query
naming a specific film ("что-то как Место встречи изменить нельзя") or actor
("фильмы с Хабенским") can miss the exact record entirely, because nothing in
the vector space privileges the literal name. Full-text search matches names
and understands nothing, which is the complementary failure mode. Neither
channel subsumes the other.

**What is indexed**: `serial_name` at weight A, `director` and `actors` at
weight B. `description` is deliberately excluded -- it is long free text that
would dominate the index by token count and match on ordinary vocabulary, and
it is exactly what the semantic channel already handles. Keeping it out
preserves the split: lexical finds names, semantic finds meaning.

**Text search config**: `russian` (snowball stemmer + stopwords). Russian is
heavily inflected, so stemming is what lets a query for "Место встречи" match
the catalogued "Место встречи изменить нельзя".

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

Three signals combined via weighted sum, each min-max normalized across the
candidate set first:

| Signal | Default Weight | What it measures |
|--------|---------------|------------------|
| Semantic | 0.4 | Cosine similarity between query and movie embeddings |
| Metadata | 0.3 | Genre overlap between LLM-extracted intent and movie genres |
| Session | 0.3 | Cosine similarity between session preference vector and movie |

**Cosine similarity mapping**: Raw cosine similarity ranges [-1, 1].
Mapped to [0, 1] via `(sim + 1) / 2` so negative similarity contributes 0,
not negative weight.

**Why normalize before summing**: the raw signals have very different spreads.
Semantic and session cosines cluster in a narrow band (roughly 0.50-0.80 in
practice), while genre overlap spans the full 0-1 range and moves in large
discrete jumps. A signal's real influence on the ranking is `weight * spread`,
not weight alone, so un-normalized the metadata signal at 0.3 outranked the
semantic signal at 0.4 -- the declared weights did not describe the actual
behaviour. Min-max normalizing each signal across the candidate pool makes
every signal span the same range, so influence equals the weight as written.

A signal identical across every candidate carries no ranking information and
collapses to a neutral 0.5 rather than being stretched across the full range
by numerical noise. Pre-normalization values are retained per candidate under
`raw_scores` for debugging and evaluation.

This is deliberately a different technique from the RRF used to fuse the two
retrieval channels above. Retrieval fuses incomparable scores across
independent systems, where only rank is trustworthy. Scoring ranks one shared
pool with three commensurable signals, where magnitude is real information
worth keeping.

**Weight tuning**: Grid search across 6 weight configurations showed minimal
impact on LLM-judged relevance (3.55-3.60 out of 5.0). Semantic similarity
dominates regardless of weight allocation. This suggests the embedding quality
is the bottleneck, not the scoring formula.

**Configurable**: weights live in `params.yaml` under `scoring.weights` and
must sum to 1.0. That file is the only place they can be changed.

## Cross-Encoder Reranking

**Choice**: `BAAI/bge-reranker-v2-m3` on the GPU, applied to the top 20 scored
candidates, blended 50/50 with the scorer total.

**Why rerank at all**: the scorer is a bi-encoder comparison -- query vector
against movie vector, each computed without ever seeing the other. A
cross-encoder reads both together in one forward pass, which is a strictly
richer view and typically improves ordering more than any weight tuning on
the existing signals. It is also what makes the lexical channel pay off:
an exact title match scores highly on direct relevance even when the
embeddings disagreed.

**Model and device are one decision**: bge-reranker-v2-m3 is an
XLM-RoBERTa-large backbone at ~302M body parameters, about 155 GFLOPs per pair
at 256 tokens. Twenty candidates is ~3.1 TFLOPs -- roughly 0.15s on this GPU
and roughly 15s on CPU. The same model is either comfortably interactive or
completely unusable depending on where it runs, so `reranking.device: auto`
(cuda when present) is not an optimisation here, it is what makes the choice
viable at all.

On a machine without a usable GPU, do not simply move this model to CPU.
Either switch `reranking.model` to a small multilingual cross-encoder such as
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (~21M body parameters, ~14x less
compute), or set `reranking.enabled: false`.

**VRAM**: ~1.1GB at fp16, on top of ~5GB for the LLM and ~1.1GB for the
embedder -- about 8.2GB of the 11GB card.

**Why blend rather than replace**: the cross-encoder judges query-document
relevance and knows nothing about session preference or extracted genre
intent. Replacing the scorer total would discard personalization entirely.
Both sides are min-max normalized across the slice before blending, so
`reranking.weight` means what it says.

**Why only 20**: every candidate is a model forward pass, so `reranking.top_k`
is the primary latency dial. Twenty is comfortably more than the five results
returned, while keeping the added latency near a second on CPU.

**Fail-safe**: a model that fails to load, or a prediction that raises, logs
and returns candidates in scorer order. A failed download must not break
recommendations, and the failure is cached so it is not retried per request.

**Not yet measured**: whether reranking improves results here is unverified.
It is a well-established technique and the reasoning is sound, but the
golden-set evaluation is the thing that would actually prove it. Treat
`reranking.weight` and `reranking.top_k` as untuned defaults until then.

## MMR Diversification

**Choice**: Maximal Marginal Relevance (Carbonell & Goldstein, 1998)

**Why**: Without diversification, top-5 results for "хочу триллер" might
return 5 movies by the same director or from the same country. MMR ensures
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

**Why EMA over simple averaging**: Simple averaging gives equal weight to all
turns. An early exploratory query permanently dilutes the signal from a later
specific query. EMA makes recent queries dominate.

**Explicit preferences**: Alongside the implicit vector, the session tracks
liked genres, disliked genres, themes, and reference films extracted from
parsed intent.

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

With default weights (0.4/0.3/0.3) and LLM-as-judge:

| Metric | Value | Interpretation |
|--------|-------|---------------|
| LLM Relevance | 3.59/5 | Honest score. System returns genre-correct movies but not always the best thematic matches. |
| NDCG@5 | 0.95 | Best-scored movies are ranked higher (strong position-aware quality). |
| Diversity | 0.39 | Results are reasonably spread across embedding space. |
| Novelty | 0.83 | Recommending non-obvious items, not just popular ones. |
| Negation Violations | 0 | Hard filter works perfectly. |

**Notable query-level results**:
- "советское кино" (Soviet cinema): LLM=5.0 (perfect, era-specific queries work well)
- "аниме для взрослых" (adult anime): LLM=1.8 (worst, age_rating not used in scoring)
- "хочу драму, но не мелодраму": LLM=2.4 (negation works but remaining dramas lack thematic fit)

**Weight sweep finding**: Minimal impact across configs (3.55-3.60). Semantic
similarity dominates. Future improvement should focus on embedding quality
(description augmentation, fine-tuning) rather than weight optimization.

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
