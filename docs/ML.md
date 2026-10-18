# ML Design Decisions and Evaluation

This document explains the reasoning behind each ML decision in the
recommendation pipeline. It covers what was chosen, what was considered,
and what the evaluation results show.

The decisions themselves are summarised, one short record each, in the
[ADR index](adr/README.md). The catalog the system searches is described in the
[catalog datasheet](datasheet.md).

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
model changed at once). Measure BGE-M3 on its own merits with the golden-set
evaluation (see Evaluation Framework) once its draft labels are reviewed.

**Embedding text**: Each movie is embedded as
`"{serial_name}. {genres joined}. {description[:N]}. Keywords: {keywords[:M] joined}"`,
built by `core.embedding_service.build_movie_embedding_text` with `N` =
`embedding.text.description_chars` (500) and `M` = `embedding.text.max_keywords` (10).
This captures title, genre signal, semantic content, and curated theme tags in one vector.
The wording/format is versioned by the `EMBEDDING_TEXT_VERSION` constant next to the
function: bump it on any template change.

**Index provenance**: the single-row `EmbeddingIndex` table records what built the
vectors -- `embedding.model`, `embedding.dimensions`, `EMBEDDING_TEXT_VERSION` and the
`embedding.text.*` values. `generate_embeddings` writes it;
`check_embedding_index` (a step of the backend start-up chain) compares it with
`params.yaml` and stops the container on a mismatch, because same-width vectors from a
different model or text are not comparable with query vectors and nothing else notices.
`generate_embeddings` refuses to embed only the missing rows under a mismatch (that would
mix vector spaces); `--all` re-embeds the whole catalog and refreshes the record. An index
built before the record existed is "legacy": `generate_embeddings --adopt` records the
current config without re-embedding (you assert the vectors were built with it).

`embedding.revision` (the Hugging Face commit the weights are downloaded at) is part of
the record too, with one difference from the other fields: a revision is compared only
when both the recorded and the configured value are set. A null on either side (an index
recorded before pinning existed, or `revision: null` = unpinned) is reported as a warning
by `check_embedding_index` and does not stop start-up or `generate_embeddings`; two
different pins are a mismatch like any other. `generate_embeddings --all` records the
configured revision.

> **Changing `embedding.model` or `embedding.dimensions` is not a config tweak.**
> `dimensions` is baked into the `Movie.embedding` pgvector column, so either change requires a schema migration *and* a full
> re-run of `manage.py generate_embeddings` over the entire catalog. Existing
> vectors are not convertible between models or dimensions -- skipping the
> re-run leaves stale vectors in place that fail comparisons silently rather
> than raising an error. The provenance record above catches the model, dimensions and
> text settings, but the dimension change itself still needs the migration. See `CLAUDE.md`.

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
answer. That breaks JSON intent parsing and adds seconds to each of the
two sequential LLM calls per turn (classify+parse, then explanation). The
flag is sent to Ollama on every request, and `ollama_client` additionally
strips reasoning spans from both parsed JSON and the streamed output, so an
Ollama build that ignores the flag degrades to slower rather than broken.

**Two roles**:
1. **Classification + intent parsing**: one combined call (`aclassify_and_parse`)
   returns the message category plus genres, mood, themes, negations, reference
   films, country exclusions/inclusions, min vote average, min/max release year,
   min/max runtime, and original languages, validated against a Pydantic schema
   with a repair retry on invalid JSON. Sends a JSON schema derived from the Pydantic
   model as Ollama's `format` (structured outputs), so decoding is constrained to the
   schema; validation and the repair retry stay as the guard.
2. **Explanation generation**: writes explanations of why each recommended
   movie matches the query (RAG pattern).

**Entity resolution** (`core/entity_resolution.py`): genres, negations,
country inclusions/exclusions and original languages are compared with the
catalog as exact strings, and the LLM often names them differently ("sci-fi"
for "Science Fiction", "USA" or "Korea" where the catalog says "United States
of America" or "South Korea"). A near-miss silently loses a filter or empties
the result. Each value goes through: exact match (casefold, collapsed
whitespace) -> short synonyms (`intent.resolution.<field>.synonyms`, targets
checked against the catalog on first use) -> embedding nearest catalog value,
accepted only if cosine >= `match_threshold` and top1 - top2 >= `match_margin`
(genres and countries only; ISO language codes take exact and synonyms only)
-> otherwise dropped with an INFO log. Genre values come from `CATALOG_GENRES`,
countries and languages from the DB (distinct values, cached per process). The
thresholds (0.5 / 0.05) are uncalibrated; `manage.py entity_resolution_report`
prints the probes in `data/entity_probes.json` with their resolution and
top-2 scores so they can be set from measurements.

**Observability**: each Ollama call logs token counts, Ollama's own prompt/eval
timings and (streamed calls) the client-side time to first token, and the digest of
the served model is logged once per process -- the `qwen3:8b` tag is mutable and cannot
be pinned by digest in `ollama pull`. A warning fires when a prompt reaches 90% of
`llm.num_ctx`. Details in [core.md](core.md); what is and is not pinned in
[infrastructure.md](infrastructure.md#reproducibility).

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

**Term extraction tunables** (`retrieval.lexical` in `params.yaml`,
applied in `candidate_generation.py`'s `_lexical_terms`): the query text
(see below) is lowercased and split into words before being turned into a
`tsquery`.

| Parameter | Value | What it controls |
|-----------|-------|-------------------|
| `min_term_length` | 3 | Drops any extracted word shorter than this many characters -- filters out prepositions/articles ("a", "of", "the") that would otherwise pollute the OR-ed query with noise terms. |
| `stopwords` | movie, like, show, ... | Conversational words that name no film but match titles and keywords ("Movie", "Like", "Show"). Dropped before the `max_terms` cut; if nothing is left the lexical channel is skipped and semantic search carries the query. |
| `max_terms` | 12 | Caps the number of terms kept (first N after the length filter), bounding `tsquery` size for a long, rambling conversational message. |

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
| Semantic | 0.4667 | Cosine similarity between query and movie embeddings |
| Metadata | 0.3333 | Genre overlap between LLM-extracted intent and movie genres |
| Popularity | 0.2 | TMDB's popularity metric (a scoring signal, not a hard filter -- it's unbounded and non-linear, unlike vote_average's bounded 0-10 scale) |

The weights are the old ones with the `session` signal removed ([ADR 0010](adr/0010-drop-session-vector-contextual-query-text.md)):
semantic and metadata keep their 7:5 proportion and share `1 - popularity`.

**Cosine similarity mapping**: Raw cosine similarity ranges [-1, 1].
Mapped to [0, 1] via `(sim + 1) / 2` so negative similarity contributes 0,
not negative weight.

**Why normalize before summing**: the raw signals have very different spreads.
The semantic cosine clusters in a narrow band, popularity is unbounded
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
re-running against the current pipeline with `evaluate_scoring --sweep` (dev
split, golden-set nDCG) once the draft labels are reviewed.

**Configurable**: weights live in `params.yaml` under `scoring.weights` and
must sum to 1.0. That file is the only place they can be changed.

## Cross-Encoder Reranking

**Choice**: `BAAI/bge-reranker-v2-m3` on the GPU, applied to the top 50 scored
candidates, blended 50/50 with the scorer total. The weights are downloaded at the
commit pinned in `reranking.revision`, so a re-download cannot change the scores the
blend and the `min_score` gate were calibrated on.

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
relevance and knows nothing about the extracted genre intent or popularity.
Replacing the scorer total would discard those signals entirely.
Both sides are min-max normalized across the slice before blending, so
`reranking.weight` means what it says.

**Why 50**: every candidate is a model forward pass, so `reranking.top_k` is
the primary latency dial. Raised from 20 to 50 once this ran on GPU: +0.38s
for a 2.5x larger reranked pool, against a ~2s LLM call already in the same
request.

**Fail-safe**: a model that fails to load, or a prediction that raises, logs
and returns candidates in scorer order. A failed download must not break
recommendations, and the failure is cached so it is not retried per request.

**"Nothing suitable" gate (`reranking.min_score`)**: retrieval always has a
top-5, so on its own it can never say "the catalog has nothing for this".
Every other signal (semantic cosine after min-max, genre overlap, popularity,
RRF rank) is relative to the rest of the pool; only the cross-encoder's raw
`rerank_score` is an absolute answer to "does this movie match this query".
`core.pipeline.apply_relevance_gate` keeps candidates with
`rerank_score >= min_score` before MMR (both `ChatView` and `evaluate_scoring`
run it, because it lives in the shared `generate_and_score`). Candidates with
no `rerank_score` ranked below the reranked head and are dropped too, so a
gated reply can have fewer than `top_n` movies or none; an empty one makes the
chat answer with a fixed reply that names the active hard filters ("Nothing in
the catalog matches all of: released 1950–1959, under 20 min. Which one should
I loosen?", or a generic "rephrase or loosen" when no hard filter was set),
skip the explanation LLM call, and report `no_match: true` in the `movies` event
(see `api.md`). The same reply is sent whenever the turn ends with no movies, so
also when hard SQL filters leave no candidates with the gate off.

**Constraints the catalog cannot check** (ADR 0011): actors, director, silent or
black-and-white, awards and studio are not in the catalog. Classify+parse puts
such conditions in `unsupported_constraints` (not a filter, kept out of the
intent dict and of `standalone_query`). A turn with any of them is not searched:
the chat asks, with a fixed reply, whether to search without them, and the next
turn's "yes" searches even if the classifier lists them again.

- *Range.* `rerank_score` is the sigmoid of the model's logit: in the pinned
  sentence-transformers line (`<3`, checked in 2.7.0) `CrossEncoder` defaults
  `default_activation_function` to `nn.Sigmoid()` for a single-label model, and
  bge-reranker-v2-m3 is single-label. So scores are in [0, 1] and
  `min_score` must lie strictly inside (0, 1) (settings refuses anything else).
  The model could not be downloaded on the build host; the calibration report
  prints the observed min/max, so confirm they fall in [0, 1].
- *Fail open.* If the reranker is disabled, did not load, failed to predict, or
  had fewer than two candidates, no candidate has a `rerank_score` and the gate is
  skipped with one WARNING per request: an infrastructure fault must never turn
  into zero results.
- *Off by default, no number shipped.* `min_score: null`. A threshold nobody
  measured is exactly the defect found in `intent.genre_match_threshold`
  (an eyeballed `0.5`): too high silently empties answerable queries, too low
  gates nothing. The value depends on this reranker and this catalog, so it is
  calibrated, not guessed.
- *Calibrating.* On the GPU machine, with the catalog imported and golden labels
  reviewed: `python manage.py evaluate_scoring --intent oracle --gate-report`
  (dev split only; it refuses `test`, as `--sweep` does). The run is ungated and
  prints, for hits vs non-hits (golden-set membership), the `rerank_score`
  count/min/quartiles/max, then per candidate threshold the share of returned
  hits it would drop, the share of non-hits dropped, and the share of queries it
  would empty (a query empties when the best score in its whole reranked pool is
  below the threshold). The same report is in the run record's `gate_report`
  key, and each query's `recommended` entries carry `rerank_score`/`hit`. A good
  threshold drops few hits and many non-hits while emptying few queries. The
  golden set holds only answerable queries, so "queries emptied" measures the cost
  of a threshold, not the benefit on unanswerable queries -- that side (queries
  the catalog genuinely cannot serve) is not measured yet. The command never
  writes `params.yaml`; set the chosen value by hand, then report on the `test`
  split (`counters.empty_results` counts gated-empty queries in a normal run).

**Not yet measured**: whether reranking improves results here is unverified.
It is a well-established technique and the reasoning is sound, but a
golden-set evaluation run (with reviewed labels) is the thing that would
actually prove it. Treat
`reranking.weight` and `reranking.top_k` as untuned defaults until then.

## Exclusion

Before the relevance gate and MMR, `core.pipeline.exclude_candidates` drops the
movies already shown (`exclude_ids`, a set of `tmdb_id`; `ChatView` passes every
`movie_id` shown within the `session.history_max_turns` window on a `refinement` turn, nothing on a `new_search`)
and the reference films
themselves: a candidate whose `serial_name` or `original_title` equals an
`intent["reference_films"]` entry after `normalize_title` (lowercase, no
punctuation, no leading the/a/an) is never recommended. Matching is exact, so
asking for "Sherlock Holmes" removes that film but keeps "Sherlock Holmes: A
Game of Shadows". The returned candidate pool stays the full reranked pool.

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

## Query Text for Lexical Search and Reranking

The lexical channel and the cross-encoder compare text, so the text matters. They
used to get the raw last message, which on a follow-up such as "suggest something
similar" carries nothing: after "I liked The Godfather" the reranker scored
"Something Like Happiness" 0.17 and The Godfather 0.0004. The classify+parse call now
also returns `standalone_query`, the request rewritten to stand on its own ("crime
dramas similar to The Godfather"). `ChatView` and `evaluate_scoring --intent e2e`
pass it as `query_text` to `generate_candidates` (lexical channel) and
`rerank_candidates`; an empty value falls back to the raw message. The dense query
embedding still comes from `semantic_query`. Conversational stopwords
(`retrieval.lexical.stopwords`) are dropped from the lexical terms. `--intent oracle`
has no `standalone_query` and ranks on the raw golden query. See
[ADR 0010](adr/0010-drop-session-vector-contextual-query-text.md), which also records
why the session preference vector was removed (ablation: mean nDCG@5 difference
-0.0143, 95% CI [-0.0310, +0.0005], helped 1 / neutral 3 / hurt 14 of 18 multi-turn
queries).

**Explicit preferences**: the session tracks liked genres, disliked genres, themes,
and reference films extracted from parsed intent. Not currently read back into
scoring or filtering -- exposed read-only via `GET /api/sessions/<id>/`.

## Evaluation Framework

`manage.py evaluate_scoring` scores the system against a labelled golden set.
It is built so that a number it prints measures what ships:

- **Relevance is golden-set membership**, not genre overlap. The old
  heuristic counted a result as relevant if it shared a genre with the
  query's labelled genres -- exactly what the `metadata` scoring signal
  computes from the same genres, so P@k was inflated by construction.
- **The intent is parsed by the production LLM call** by default (`--intent
  e2e`), so the most fragile link -- parsing -- is evaluated too.
- **Retrieval is the production function.** `core.pipeline.generate_and_score`
  (candidates -> scoring -> reranking -> exclusion -> gate -> MMR) is what `ChatView` runs through a
  thin `sync_to_async` wrapper, so eval and production cannot drift.
- **Weights are selected on dev, reported on test**, and every run leaves a
  JSON record behind.

### Golden Set

`data/golden_set.json`: ~30 queries, each with `id`, `query`, `split`
(`dev`/`test`), and `relevant` -- movie references as `{"tmdb_id": int}` or
`{"title": str, "year": int}`, optionally with `grade` (1 relevant, 2 highly
relevant; default 1). Optional fields: `negated_genres` (for the
negation-violation check), `oracle_intent` (the intent `--intent oracle` uses),
`context` (prior user messages, for multi-turn entries) and `category`
(`new_search`/`refinement`, the final message's category in oracle mode).

**The labels are a draft.** They were written from general film knowledge
without catalog access, as `{title, year}` references, and have not been
reviewed by a human. At run time each reference is resolved against `Movie`
(title matched case-insensitively on `serial_name` or `original_title`, plus
the `release_date` year; `tmdb_id` directly). Unresolved references are
listed per query on stdout and in the run record (`unresolved_refs`); they
drop out of that query's recall/nDCG denominator, visibly. A query with no
resolved reference at all is reported as unscored (`unscorable_queries`)
rather than counted as 0. Lists are a sample of good answers, not
exhaustive, so precision on broad queries ("a good comedy") is a lower bound.

The loader and resolver live in `core/golden_set.py`, separate from the
command, so other evaluations can reuse them.

**Splits.** ~2/3 `dev`, ~1/3 `test`, both containing negation and multi-turn
entries. `--sweep` only ever selects on `dev` (it refuses `--split test` and
`--split all`); the plain eval defaults to `test`. Picking weights on the set
you then report on overstates the result.

### Intent Modes

| Mode | Intent | Query embedding | Use it for |
|------|--------|-----------------|------------|
| `e2e` (default) | `aclassify_and_parse(message, history)`, as `ChatView` calls it | `encode_query(intent.semantic_query)` | What users get |
| `oracle` | The entry's `oracle_intent` (`{}` if absent) | `encode_query(query)` | Retrieval in isolation from parsing |

In `e2e` the prior `context` messages are passed as user-role history entries
(the shape `ChatSession.history` uses), and the predicted category is
recorded per query. A golden query classified `follow_up`/`general_chat` is
still searched, so it still gets metrics, but it counts in `misclassified`
-- in production it would have produced no recommendations.

`aclassify_and_parse` never raises: on a failed request or an invalid reply
it falls back to `new_search` with the raw message and no filters. So `e2e`
refuses to start when Ollama is unreachable (`ais_available()`), and any
parsed intent with exactly that fallback shape is flagged
`possible_parse_fallback` and counted in `possible_parse_fallbacks` -- a
vague query can genuinely parse that way, but a run where many do is a run
where parsing failed. A parsed intent with `unsupported_constraints` (ADR
0011) is still searched, but its row is flagged `UNSUPPORTED(...)` and it
counts in `unsupported_constraints`: in production that turn asks a question
and shows no movies. The golden set holds user turns only, so the exclusion
of already-shown movies on refinements is not reproduced. Comparing `e2e`
against `oracle` on the same split separates parsing errors from retrieval
errors (oracle intents are labels too, so they are only as good as the
draft).

### Metrics

| Metric | What it measures | Implementation |
|--------|-----------------|----------------|
| Precision@k | Fraction of the top k that are golden movies | `evaluation.precision_at_k()` |
| Recall@k | Fraction of the resolved golden movies found in the top k | `evaluation.recall_at_k()` |
| nDCG@k | Position-aware, graded: DCG uses each result's golden grade (0 if not golden), IDCG the golden grades sorted descending, top k | `evaluation.ndcg_at_k()` |
| Diversity | 1 - mean pairwise cosine similarity among results | `evaluation.diversity()` |
| Novelty | How non-obvious the recommendations are (genre rarity) | `evaluation.novelty()` |
| Coverage | Fraction of catalog appearing in any recommendation | `evaluation.coverage()` |
| Negation Violations | Results containing a negated genre (safety metric, separate from relevance) | `evaluate_scoring._score_query()` |
| LLM Relevance | Mean judge score (1-5) over successfully judged results | `evaluation.llm_relevance_score()` |
| Judge nDCG | nDCG over judge scores, gain = score - 1, ideal = k results at 5/5 | `evaluation.judge_ndcg()` |

nDCG's ideal ranking comes from the labels, never from the returned list --
the old graded mode sorted the returned scores to build its ideal, so a list
of uniformly bad results scored 1.0.

### LLM-as-Judge

With `--llm-judge`, Ollama rates each result 1-5 (for multi-turn entries it
sees the whole conversation, not just the last message). It is reported as
`llm_relevance` and `judge_ndcg`, beside -- not instead of -- the golden-set
metrics. A failed judge call (request error, unparseable reply, score off the
scale) returns `None`: it is skipped in the mean, counted in
`judge_failures`, and a query with any failure gets no `judge_ndcg`. There is
no substitute score; the old code silently returned 3. Its description cap
and timeout are `evaluation.judge_description_chars` /
`evaluation.judge_timeout_seconds` in `params.yaml`.

**Known limitation (circularity)**: the judge is the same model family that
parses intent and writes explanations, so it shares the system's blind
spots. `--sweep` ranks configurations by golden-set nDCG, never by the judge.
The swept semantic and metadata weights (and `--weights`, which takes `semantic,metadata`) are scaled to
`1 - popularity`, with popularity held at its configured value, so every printed or
recorded configuration sums to 1.0 and the best one can be pasted into
`scoring.weights`.

### Determinism

Every Ollama call sends `options.seed` (`llm.seed`) and a temperature chosen
by call kind: `llm.temperature.structured` (0.0) for every schema-constrained call --
classify+parse, the sync intent call, the judge -- and
`llm.temperature.generation` (0.7) for explanations and conversational
replies. With greedy decoding on the structured calls, `e2e` intent parsing
and the judge should give the same output for the same input (on the same Ollama
build and hardware; GPU kernels may still vary the result), so two runs of the
same code and params are comparable.

### Run Records

Every run writes a JSON record to `--output`, default
`eval_runs/<eval|sweep|gate_report>-<UTC timestamp>.json` under the working directory
(`eval_runs/` is git-ignored; it is not under `data/`, which is mounted
read-only in Docker). In the Docker setup the backend container's `/app/eval_runs`
is bind-mounted to `./eval_runs/` at the repo root, so records land there on the
host and survive `docker compose down`/`up --build`. It holds the git commit and dirty flag (`null` when git
or `.git` is unavailable: always the case for a run inside the Docker image, so
the commit of a Docker run is not recorded -- note it yourself if it matters), the sha256 of `params.yaml`
and of the golden set, the CLI options, intent mode, split, k, weights, a
timestamp, per-query results (recommended ids/titles, hits, resolved and
unresolved references, predicted category in `e2e`, judge scores), aggregate
metrics, and counters (`unresolved_refs`, `unscorable_queries`,
`judge_failures`, `misclassified`, `possible_parse_fallbacks`, `unsupported_constraints`,
`empty_results`). A sweep records every configuration this
way, sorted best first. Stdout stays a human-readable summary.

### Running Evaluation

```bash
# Production path (e2e intent) on the held-out test split
docker compose exec backend python manage.py evaluate_scoring

# Retrieval only: labelled intent instead of the LLM parse
docker compose exec backend python manage.py evaluate_scoring --intent oracle

# Another split (dev | test | all)
docker compose exec backend python manage.py evaluate_scoring --split dev

# Plus LLM-as-judge (slow, ~2s per movie; secondary signal)
docker compose exec backend python manage.py evaluate_scoring --llm-judge

# Grid search over weight space (dev split only; intent is parsed once per query)
docker compose exec backend python manage.py evaluate_scoring --sweep

# Grid search with LLM judge (6 configs)
docker compose exec backend python manage.py evaluate_scoring --sweep --llm-judge

# Calibrate reranking.min_score (dev split only; gate off for the run; prints the report)
docker compose exec backend python manage.py evaluate_scoring --intent oracle --gate-report

# Custom weights, explicit run-record path
docker compose exec backend python manage.py evaluate_scoring --weights 0.6,0.2,0.2 --output eval_runs/custom.json
```

### Evaluation Results

**None yet.** No run of the golden-set evaluation has been recorded. The
numbers that used to live here were measured with the old genre-overlap
heuristic (inflated by construction, see above) against the old Okko catalog
(18,130 items), the old mpnet embedder (768-dim), the old 3-signal weight
scheme, and a Russian-language test set -- none of them describe the current
system. Before reporting numbers here: import and embed the catalog, review
the draft labels (start with whatever the run reports as unresolved), then
run on `test` with weights chosen on `dev`, and cite the run record.

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
