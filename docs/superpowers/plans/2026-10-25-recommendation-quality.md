# Recommendation Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

## First: merge the inference modes into dev

- [ ] In the existing `dev` worktree, merge `feat/local-openrouter-modes` into `dev`. Inspect the working trees and branch histories first, preserve existing work, and resolve conflicts without dropping either the completed quality tasks or the two-mode architecture.
- [ ] Run the offline backend test suite with external networking blocked and HTTP adapters mocked. Inspect any failures before proceeding; do not make real API requests or start the application as an agent.
- [ ] Continue all tasks below on `dev`; do not push. Tasks 1–4 are already implemented, but Task 4's user-reviewed `data/label_queries.json` must exist before the export.

**Goal:** Recommend movies that are both relevant and good, not obscure low-rated filler, except when the user explicitly asks for niche or trashy films. All tuning must be learned from the user's own graded labels.

**Architecture:** Two-stage retrieval → ranking stays as it is. Three changes on top of it, plus one tuning pass:
1. A quality signal (IMDb-style Bayesian weighted rating) enters the scorer.
2. A third retrieval channel puts well-known films into the candidate pool.
3. One LLM-extracted flag (`wants_obscure`) turns both off.

Weights are fitted offline on CPU from user grades collected with pooling. The queries come from Reddit-Movie. Generate the initial candidate pool locally on the user's GPU when available; reuse the same grades for matching OpenRouter query/movie pairs. Run OpenRouter separately to obtain its own scores and label only additional candidates as needed.

**Tech Stack:** Django, Postgres/pgvector, numpy, local Ollama and sentence-transformers, or OpenRouter adapters for LLM, embeddings and reranking. No new dependencies.

**Spec:** no separate spec document. The source is the brainstorm in this session, 2026-10-25. Decisions:
- quality before obscurity;
- niche only on explicit request, and then the quality signal is off rather than inverted;
- no hardcoded per-scenario rules and no continuous LLM "slider";
- queries come from an existing dataset, Reddit-Movie;
- the user labels "would I recommend this" on a graded scale and is ready to label a lot.

## Evidence this plan argues from

**The user's 300 labels joined with the TMDB catalog:**
- 47% of returned movies have fewer than 100 votes (the catalog median is 59);
- 30% are rated below 6.0;
- only 9% are "known and good" (≥1000 votes and ≥7.0).

**Code at the start of Tasks 1–4:** no quality term in `core/scoring.py` (semantic 0.4667, metadata 0.3333, popularity 0.2). The existing 300 binary labels are relevance judgments, not graded "would recommend" judgments. Keep them as binary relevance evidence and as source material for optional manual regrading; never automatically convert a binary 1 into a quality grade.

**Industry and papers:**
- Two-stage retrieval → ranking is the standard architecture ([YouTube DNN](https://research.google/pubs/pub45530)).
- Production systems blend several retrievers ([Shaped](https://shaped.ai/blog/anatomy-of-modern-ranking-architectures-part-2)).
- Popularity and quality features help learned rankers but can dominate relevance features, so keep relevance as a gate ([arXiv 1903.04263](https://arxiv.org/pdf/1903.04263), [arXiv 2410.05763](https://arxiv.org/pdf/2410.05763)).
- Bayesian weighted rating, `WR = v/(v+m)·R + m/(v+m)·C`, as used for IMDb's Top 250 ([write-up](https://neurobiology.substack.com/p/how-imdb-determines-top-250_27)).
- Depth-k pooling across system variants, shown to assessors in random order with a graded scale ([arXiv 2211.00981](https://arxiv.org/pdf/2211.00981), [Lin et al.](https://arxiv.org/pdf/2010.06467)).
- Retrieval-stage recall is a common bottleneck in reranked movie recommenders ([arXiv 2604.16318](https://arxiv.org/pdf/2604.16318)).
- Extracting a structured preference slot from the query with an LLM is current industry practice ([Instacart](https://tech.instacart.com/building-the-intent-engine-how-instacart-is-revamping-query-understanding-with-llms-3ac8051ae7ac), [AACL 2025](https://underline.io/lecture/138086-hint-augmented-re-ranking-efficient-product-search-using-llm-based-query-decomposition)).
- Query source: [Reddit-Movie](https://huggingface.co/datasets/ZhankuiHe/reddit_movie_large_v1) from [He et al. 2023](https://arxiv.org/pdf/2308.10053), IMDb-linked and research-use only. Fallback: ReDial, CC BY 4.0.

## Historical hardware trade-off (GTX 1080 Ti, 11 GB, Pascal sm_61)

The following table is historical reasoning for the earlier Qwen3 8B profile, not a measurement of the current two-mode configuration or a cloud latency estimate. Current local Qwen3.5 9B GPU fit remains unverified; follow `docs/infrastructure.md` after merging the inference-modes branch.

| Option | GPU cost | Latency | Verdict |
|---|---|---|---|
| Quality signal in the scorer | 0 | ~0 (one cached aggregate) | **do** |
| Quality retrieval channel | 0 (Postgres exact cosine over a ~10% subset) | +tens of ms | **do** |
| `wants_obscure` intent flag | 0 VRAM, ~+50 prompt tokens of ~0.6k headroom | 0 extra calls | **do, measure tokens** |
| Weights fitted offline on labels (numpy, CPU) | 0 | 0 at serve time | **do** |
| Bigger reranker / second cross-encoder | +1–2 GB | +0.5–1 s | no: no VRAM |
| LLM listwise rerank of top-k (qwen3:8b) | 0 VRAM, but KV cache and window | +2–4 s per turn, order-sensitive | no (also ADR 0001) |
| Fine-tuning bge-reranker | training on the 1080 Ti | 0 | no: too little data for now |
| LambdaMART (lightgbm, CPU) | 0 | ~0 | later, only if linear weights plateau with >2k labels |

## Global Constraints
- Every tunable value lives in `params.yaml`. No env vars, no literals in code (CLAUDE.md).
- New weights ship at a value that leaves behaviour unchanged (`scoring.weights.quality: 0.0`, `retrieval.rrf_weights.quality: 0.0`) until Task 7 sets fitted values.
- `scoring.weights` must still sum to 1.0 (`_validate_score_weights` in `backend/recommender/settings.py:257`).
- Preserve the merged branch's canonical embedding model, dimensions, text template and provenance; never rebuild embeddings as part of this plan. Respect its compatibility check and recovery instructions.
- `wants_obscure: true` turns off the quality signal and the quality channel, nothing else. The quality signal is never inverted.
- The LLM judge is not ground truth (CLAUDE.md). Only the user's grades count.
- Grades: `0` would not recommend, `1` acceptable, `2` good pick, `3` great pick, judged against the query text.
- Agents never perform real API calls, model downloads, app startup or real GPU checks; use mocked adapters and block external networking in tests. Only the user runs GPU, Docker startup, real API and compatibility checks. Do not read or edit `.env`.
- Local launch: `docker compose up --build`. OpenRouter launch: `docker compose -f docker-compose.yml -f docker-compose.openrouter.yml up --build`; use the same prefix for `exec`. Switching modes and an existing index follow the merged `docs/infrastructure.md`, with no automatic rebuild. Host tests: `cd backend && uv run pytest`.
- Initial export runs locally on GPU when available. The current machine has no GPU, so implementation and mocked verification can proceed now; the local export, grading and evidence-dependent tuning wait for the user's GPU run.
- Keep user grades independent of mode and raw model scores. Fit or transfer scoring weights only after measurement; calibrate BGE and Voyage absolute relevance thresholds separately. No provider fallback, and no branching by mode inside the shared pipeline.
- Commits: one per logical part, no Claude attribution, never push.
- Preserve existing evaluation flags and datasets until the replacement is accepted for both modes. Task 8 removes only obsolete functionality after that point; the golden set remains a behaviour check.

## Review Focus
1. A movie with `vote_count` 0 or `vote_average` 0.0 (model defaults) gets WR = C, a neutral value: no zero, no crash. Owned by Task 1.
2. Hard filters leave the quality channel's subset empty (for example "1950s Korean documentaries"). Candidate generation falls back to the other channels without error. Owned by Task 2.
3. `wants_obscure` together with `reference_films` or a `min_vote_average` filter: the flag only disables quality, and the explicit rating filter still applies. Owned by Task 3.
4. An empty catalog makes `catalog_mean_rating()` return None. The quality signal collapses to `neutral_score` for the whole pool. Owned by Task 1.
5. An invalid or conflicting graded row fails with its CSV line; blanks remain unjudged, not grade zero, and incomplete comparisons are reported rather than shipped. Owned by Task 6.
6. An interrupted or failed provider call cannot produce a successful run; resume rejects changed query set, mode, config or catalog. Owned by Task 5.
7. Local and cloud exports reuse matching human grades but never mix provider scores or gate thresholds. Owned by Tasks 5–7.

---

### Task 1: Quality signal (Bayesian weighted rating) in the scorer

**Files:**
- Modify: `backend/core/scoring.py` (`SIGNALS`, `score_candidates`, new helpers)
- Modify: `backend/core/catalog_bounds.py` (cached catalog mean rating, cleared by the existing `clear_catalog_year_cache`, renamed to `clear_catalog_cache` with all callers updated)
- Modify: `backend/recommender/settings.py` (load and validate `scoring.quality.prior_votes` > 0; weights validation now covers `quality`)
- Modify: `params.yaml`: `scoring.weights.quality: 0.0` (others unchanged), new `scoring.quality.prior_votes: 500` (starting value, refitted in Task 6/7), with a comment in the file's style citing the IMDb formula
- Test: `backend/core/tests/test_scoring.py`, `backend/core/tests/test_catalog_bounds.py`

**Interfaces:**
- Produces: `catalog_mean_rating() -> float | None` (mean `vote_average` over movies with `vote_count > 0`, cached like `catalog_year_range`).
- Produces: `weighted_rating(vote_average: float, vote_count: int, prior_votes: float, catalog_mean: float) -> float`.
- Produces: `SIGNALS = ("semantic", "metadata", "popularity", "quality")`; `raw_scores["quality"]` holds the pre-normalization WR.
- Produces: `effective_weights(weights: dict, intent: dict) -> dict`. When `intent.get("wants_obscure")` is true, it sets `quality` to 0 and renormalizes the rest to sum 1; otherwise it returns the weights unchanged. `score_candidates` uses it.

- [ ] Step 1: failing tests:
  - `test_weighted_rating_shrinks_low_vote_movies_to_catalog_mean`: `weighted_rating(9.0, 10, 500, 6.3)` is within 0.06 of 6.3; `weighted_rating(8.0, 50000, 500, 6.3)` is within 0.02 of 8.0.
  - `test_weighted_rating_zero_votes_is_catalog_mean`: `(0.0, 0, 500, 6.3) == 6.3`.
  - `test_quality_signal_neutral_without_catalog_mean`: with `catalog_mean_rating` patched to None, every candidate's `quality == settings.NEUTRAL_SCORE`.
  - `test_effective_weights_drop_quality_for_obscure`: `{"semantic":0.4,"metadata":0.3,"popularity":0.1,"quality":0.2}` with `{"wants_obscure": True}` gives quality 0 and the rest summing to 1.0 in their 4:3:1 proportion.
  - `test_quality_weight_zero_keeps_order`: with shipped params, `total` ordering equals the pre-change ordering on a fixture pool.
- [ ] Step 2: `uv run pytest core/tests/test_scoring.py -q` → FAIL.
- [ ] Step 3: implement the interfaces above. `_quality_score(movie)` reads `vote_average` and `vote_count` from the candidate dict (already present, `candidate_generation.py:247-249`).
- [ ] Step 4: tests pass; full host suite shows no new failures.
- [ ] Step 5: commit `feat: add a Bayesian weighted-rating quality signal to the scorer`.

### Task 2: Quality retrieval channel

**Files:**
- Modify: `backend/core/candidate_generation.py` (`generate_candidates`, new `_quality_ranked_ids`, docstring diagram)
- Modify: `params.yaml`: `retrieval.rrf_weights.quality: 0.0`, new `retrieval.quality_channel.min_vote_count: 1000`, with a comment on why the subset exists (good films the semantic top-100 misses)
- Modify: `backend/recommender/settings.py` (load and validate `min_vote_count` ≥ 0)
- Test: `backend/core/tests/test_candidate_generation.py`

**Interfaces:**
- Consumes: `intent["wants_obscure"]` (Task 3 adds it to the schema; until then the key is absent and the channel acts on its weight alone).
- Produces: `_quality_ranked_ids(base_queryset, query_embedding: list[float], limit: int, min_vote_count: int) -> list[int]`. This is the same exact-cosine ordering as the semantic channel over `base_queryset.filter(vote_count__gte=min_vote_count)`, so hard filters still apply.
- Produces: `generate_candidates(..., quality_channel: bool | None = None)`. None means on iff `settings.RRF_WEIGHTS["quality"] > 0` and not `wants_obscure`. Task 5's export passes True to force it.

- [ ] Step 1: failing tests (mock the querysets the way the file's tests already do):
  - `test_quality_channel_skipped_when_weight_zero`;
  - `test_quality_channel_fused_under_its_own_rrf_weight`: a film present only in the quality list appears in the output;
  - `test_quality_channel_skipped_for_obscure_intent`;
  - `test_quality_channel_empty_subset_falls_back`.
- [ ] Step 2: run → FAIL. Step 3: implement. Step 4: run → PASS plus the full host suite.
- [ ] Step 5: commit `feat: add a well-known-films retrieval channel fused by RRF`.

### Task 3: `wants_obscure` intent flag

**Files:**
- Modify: `backend/core/ollama_client.py`:
  - `MessageIntent.wants_obscure: bool = False` with a tolerant validator in the file's style;
  - one schema line in the classify prompt field list: `"wants_obscure": true only if the user explicitly asks for obscure, little-known, weird, cult-trash or low-rated films, else false`;
  - add `"wants_obscure": false` to every existing few-shot JSON line, and nothing else to the prompt;
  - keep it out of `NON_FILTER_FIELDS`, so it reaches the pipeline intent dict.
- Modify: `backend/core/pipeline.py` docstring only, if it lists intent keys.
- Modify: `params.yaml` `llm.num_ctx` comment: the new estimate is the measured max + the added tokens. Count the added prompt characters ÷ 4 and state it.
- Test: `backend/core/tests/test_ollama_client.py`

**Interfaces:**
- Produces: `intent["wants_obscure"]: bool` in the dict that `ChatView` and `evaluate_scoring` pass to `generate_and_score`, consumed by Tasks 1 and 2.

- [ ] Step 1: failing tests:
  - `test_wants_obscure_defaults_false_and_coerces_strings` ("true" → True, None → False);
  - `test_wants_obscure_survives_into_intent_dict` (`model_dump(exclude=NON_FILTER_FIELDS)` contains it);
  - `test_classify_prompt_mentions_wants_obscure_once`.
- [ ] Step 2: FAIL. Step 3: implement. Step 4: PASS plus the full host suite.
- [ ] Step 5: commit `feat: let the classifier flag an explicit request for obscure films`.

### Task 4: Label-query set from Reddit-Movie

**Files:**
- Create: `backend/core/label_queries.py` (pure selection logic, no DB)
- Create: `backend/movies/management/commands/import_label_queries.py`
- Modify: `params.yaml` new `evaluation.label_queries` block: `count: 120`, `test_share: 0.33`, `seed: 7`, `min_chars: 30`, `max_chars: 600`, `min_linked_recs: 3`
- Create (output, committed after the user's review): `data/label_queries.json`
- Modify: `docs/datasheet.md` (provenance and license note)
- Test: `backend/core/tests/test_label_queries.py`

**Interfaces:**
- Produces: `imdb_to_tmdb(catalog_csv: Path) -> dict[str, int]` (from the `imdb_id` and `id` columns of `TMDB Movie Dataset v11.csv`).
- Produces: `select_queries(records: list[dict], imdb_map: dict[str, int], count: int, test_share: float, seed: int, min_chars: int, max_chars: int, min_linked_recs: int) -> list[dict]`. Each output record is `{"id": str, "query": str, "split": "dev"|"test", "community_recs": list[int]}`, where `community_recs` holds the tmdb_ids of films that Reddit replies recommended and that exist in the catalog.
- Command: `python manage.py import_label_queries --source <downloaded Reddit-Movie file> --catalog "TMDB Movie Dataset v11.csv" --out data/label_queries.json`. It reads the downloaded file; it does not download it itself. The builder confirms the dataset's file format and license from the README and records both in `datasheet.md`. If the license forbids this use, the same interface takes ReDial's first seeker utterances, with `community_recs` from recommender-mentioned movies.

- [ ] Step 1: failing tests on a 6-record fixture:
  - length bounds respected;
  - records with fewer than `min_linked_recs` dropped;
  - deterministic for a fixed seed;
  - split share within ±1 query;
  - ids stable.
- [ ] Step 2: FAIL. Step 3: implement. Step 4: PASS.
- [ ] Step 5: commit `feat: build a label-query set from Reddit-Movie requests`.
- [ ] Step 6 (user): run the command, delete unusable queries (non-requests, non-English), then commit `data/label_queries.json`.

### Task 5: Mode-aware pooled export and reusable human labels

**Goal:** Produce the first labelling pool on the local GPU, then reuse its grades for the same query/movie pairs in the OpenRouter pool. Real GPU and OpenRouter runs are user-only.

**Files:**
- Create: `backend/core/graded_pool.py` for pooling, manifests, snapshots and grade reuse.
- Modify: `backend/core/candidate_generation.py` for an explicit RRF override and channel diagnostics; preserve production defaults.
- Modify: `backend/movies/management/commands/evaluate_scoring.py`, `params.yaml` and settings validation.
- Test: new `backend/core/tests/test_graded_pool.py`, existing candidate-generation and management-command tests.

**Interfaces:**
- Add `generate_candidates(..., rrf_weights: dict | None = None)`; pass the override into RRF, and expose channel ranks/membership to export without changing the default production result.
- Add `evaluate_scoring --queries PATH --pool-export PATH [--split dev|test|all] [--limit-queries N] [--resume] [--reuse-grades PATH]`. Default split is dev; a positive `--limit-queries` takes the first N selected queries in query-file order and is recorded in the manifest. Query files require end-to-end intent. Reject incompatible evaluation modes and prevent overwriting an existing export without `--resume`.
- Export the labelling CSV with `query_id, query, split, tmdb_id, title, year, overview, grade`; use the exact original query and any conversation context when judging. Do not reveal model scores, pool membership or source ranks to the assessor.
- Store a manifest and per-query snapshots beside each CSV under `<csv-stem>.run/`. The manifest records schema version, query-file hash, split, mode, provider/model/revision metadata, effective configuration hash, catalog/index fingerprint and pooling settings. Snapshots preserve parsed intent, rewritten query, full candidate sets and order for each retrieval variant, complete raw scoring features, catalog mean, raw reranker scores and candidate embeddings needed for MMR. Keep model scores scoped to their own run.
- CSV grade identity is the same stable query ID **plus exact original query and context, split, TMDB ID and grading rubric version**. Copy an existing 0–3 grade only when all match; conflicting grades are an error, and unmatched pairs are left blank. Do not use rewritten query or inferred intent to match grades.
- Import old `eval_runs/labels.csv` as a separate binary relevance source only. Its 300 labels cover 30 queries (139 positive, 161 negative); keep its source and context, do not silently convert `1` to a 0–3 quality grade, and allow the user to explicitly regrade a matching row. Preserve this source file.

**Pool and replay coverage:**
- For each query, obtain production e2e intent and embedding using the selected adapters, then capture retrieval with the configured quality RRF weight and with quality weight `1.0`; retain all other RRF weights. The present configured quality weight is `0.0`. Merely forcing the quality channel on with zero fusion weight does not test its contribution.
- For `wants_obscure`, disable quality retrieval and the quality scoring signal in every configuration. Preserve other explicit filters.
- Take the union of both filtered candidate sets, rerank every member using the active adapter, and persist raw results. Use `evaluation.pool_export.rerank_batch_size: 50`; compute production min-max normalization on the selected head at replay, never per export batch. Add a test that mocked batching preserves candidate-score association and report that real-provider batch equivalence is subject to the user pilot.
- Pool the top `evaluation.pool_export.depth: 10` from a small, fixed set: shipped blend, quality-weight `0.3` with remaining scoring weights renormalized, and pure reranker, for each applicable retrieval variant. Include community recommendations present in the retrieved union. Deduplicate and shuffle deterministically with `evaluation.pool_export.seed: 7`; report retrieval and rerank-head community recall per variant.
- With a null gate, do not filter on raw reranker score. Local and OpenRouter reranker scores and thresholds must never be shared.

**Reliability and incremental run:**
- Save successful snapshots atomically per query. Mark runs failed on provider failures or fallback; never publish a failed query as successful. `--resume` skips complete queries only when manifest identity matches; otherwise fail before inference. Repeating an incomplete query can incur API charges.
- Show planned query count, candidate count and pooling settings before the run; do not pretend a reliable token price or exact cost is known in advance.
- Start with a user-run local dev pilot of 5 requests to measure pool size, runtime and labelling burden. Use a separate pilot output; the full run processes the complete set and may reuse manually assigned pilot grades using `--reuse-grades`. Pilot snapshots are retained as evidence, but do not skip inference in the separate full run.

**Steps:**
- [ ] Write failing tests for RRF override, zero-weight channel, deduplication, seeded shuffle, obscure intent, grade reuse across modes, query/context/rubric mismatches, conflicting grades and old binary-label preservation.
- [ ] Add tests for incomplete-query resume, changed manifest rejection, score alignment across batches and recorded provider failures; verify the targeted tests fail.
- [ ] Implement the export, immutable run metadata and reusable grades; mock all adapters in tests and block network access.
- [ ] Run `cd backend && uv run pytest core/tests/test_graded_pool.py core/tests/test_candidate_generation.py movies/tests/test_evaluate_scoring.py -q` and the full offline backend suite; expect PASS.
- [ ] Commit `feat: export resumable mode-aware graded pools`.
- [ ] User on GPU: prepare and review `data/label_queries.json`, run the 5-query local dev pilot, review its size, then export the full local dev pool. Preserve the pilot and full run records.
- [ ] User grades the local pool on the 0–3 scale. Once cloud verification is desired, run the same query set in OpenRouter using `--reuse-grades` and grade only unmatched pairs.

**User commands:** With the local stack started by the user, use:

```bash
docker compose exec backend python manage.py evaluate_scoring --queries data/label_queries.json --split dev --pool-export eval_runs/pool-local-dev.csv
```

For the pilot use `--limit-queries 5 --pool-export eval_runs/pool-local-pilot.csv`; if pilot rows were graded, pass `--reuse-grades eval_runs/pool-local-pilot.csv` to the full run. For cloud, use the explicit cloud Compose prefix from Global Constraints and the same command with `--pool-export eval_runs/pool-openrouter-dev.csv --reuse-grades eval_runs/pool-local-dev.csv`. Use `--resume` only for the same run. If the GPU is unavailable, implementation and mocked tests proceed; the initial local data collection waits.

### Task 6: Offline grade report and modest CPU fitting

**Goal:** Compare rankings against human grades without DB access, inference, GPU or API charges; avoid treating unjudged candidates as bad films or fitting more parameters than the data supports.

**Files:**
- Create: `backend/core/graded_evaluation.py` for replay, metrics, grouped cross-validation and fitting.
- Modify: `backend/core/graded_pool.py`, `backend/movies/management/commands/evaluate_scoring.py`, `params.yaml` and settings validation.
- Test: `backend/core/tests/test_graded_evaluation.py` and management-command tests.

**Interfaces:**
- `replay_ranking(snapshot: dict, params: dict, k: int) -> list[dict]`.
- `grade_report(snapshots: list[dict], grades: dict, params: dict, k: int) -> dict`.
- `fit_ranking_params(snapshots: list[dict], grades: dict, config: dict) -> dict`.
- `evaluate_scoring --grade-report PATH [--fit] [--split dev|test]`: read the CSV and companion snapshots offline; take mode and baseline configuration from the manifest, not current `INFERENCE_MODE`. Reject `--fit --split test`.

**Replay and metrics:**
- For each retrieval variant, recompute WR using saved vote counts, ratings and catalog mean; normalize scoring signals across the **full corresponding candidate set**, compute effective weights including obscure handling, then select the production rerank head. Normalize scorer totals and raw rerank scores across that head, blend, apply reference exclusions and gate, and run production MMR on saved embeddings. Respect production tie handling and top-k. The label CSV alone is insufficient for replay.
- Graded metrics: nDCG@5 with gain `2**grade - 1`, junk@5 (grade 0), good@5 (grade >= 2), community recall, empty-result rate and judged coverage. Report ordinary and obscure queries separately; report query counts for each.
- Blanks stay unjudged. Error with CSV line for out-of-range grade, unknown pair or conflicting duplicate; skip and count queries with fewer than two grades. Never assign zero to an unjudged item.
- Require all top-5 entries of compared configurations to have grades before reporting an acceptance comparison or applying a fit. Produce a deduplicated CSV of missing pairs for the user to grade, retaining existing grades. Report incomplete coverage explicitly and do not claim complete evaluation.
- On dev, use deterministic grouped five-fold cross-validation by query ID and fixed seed `7`, without splitting one query between folds. Report baseline and selected-configuration mean/std, plus paired query-bootstrap 95% intervals with a fixed seed. If fewer than five eligible queries exist, fail with an actionable message.
- Keep the candidate grid deliberately small: baseline and quality weights `0.1, 0.2, 0.3`, with the other scoring weights scaled proportionally; evaluate each with saved quality RRF variants `0.0` and `1.0`. Hold `prior_votes`, rerank blend weight, production top-k and MMR settings at the recorded baseline. Select by dev nDCG@5, break ties by lower junk@5 and then proximity to baseline; fit the final choice on dev only. Record pool bias and the number of configurations evaluated.
- Separately calibrate the selected mode's absolute gate on dev grades only. Test `null` and observed raw scores; select the strictest threshold retaining at least one relevant graded film for every answerable query and dropping at most 5% of relevant graded films. Report coverage and require full judged comparison; otherwise leave gate uncalibrated. A calibrated BGE threshold does not apply to Voyage.
- Use snapshots for paired baseline/fitted comparisons and freeze all choices before inspecting test grades. A completed test report is not a new tuning opportunity.

**Steps:**
- [ ] Write failing tests for full-pool normalization, changing rerank head, MMR and exclusions, obscure intent and null gate; compare replay against actual pipeline output using mocked local and cloud adapters.
- [ ] Add hand-calculated metric, CSV validation, grouped fold, unjudged top-5, no-network/no-DB and test-fit rejection cases. Verify FAIL.
- [ ] Implement offline replay, report, fitting and gate table using numpy and existing shared scoring helpers.
- [ ] Run `cd backend && uv run pytest core/tests/test_graded_evaluation.py movies/tests/test_evaluate_scoring.py -q` and the full offline backend suite; expect PASS.
- [ ] Commit `feat: fit and report graded rankings offline`.

### Task 7: Validate and apply settings separately for local and OpenRouter

**Files:** `params.yaml`, `docs/ML.md`, `docs/core.md`, `docs/infrastructure.md`, `docs/adr/README.md`, a new quality-calibration ADR using the next available number, and settings/reranking tests.

**Execution and acceptance:**
- [ ] User runs the dev report/fit on the graded local GPU snapshot. Review pool coverage and labelling burden; reuse the same query/movie grades for the OpenRouter dev export and label only new pairs.
- [ ] Compare local-fitted scoring weights against the cloud baseline on the cloud dev snapshot. If shared weights perform adequately, retain them; use cloud-specific scoring overrides only if cloud dev evidence supports the change. Keep all tunables in `params.yaml` and cloud-specific overrides under `runtime_profiles.openrouter`.
- [ ] Calibrate local BGE and cloud Voyage absolute relevance gates independently from each mode's own dev scores and human grades. Voyage's existing `min_score: null` remains uncalibrated until supported by cloud evidence.
- [ ] Freeze each mode's selected settings before test grading. User exports both modes on the same held-out test queries and reuses matching grades; only additional pairs require grading. No test-based tuning.
- [ ] Compare each mode's fitted configuration with its frozen baseline on the same mode-specific test snapshots. Require complete judged top-5, lower junk@5 and no reduction in nDCG@5; report query-bootstrap intervals, empty-result rate, label coverage and sample size. If the criteria fail, retain the baseline, document the result and do not claim acceptance.
- [ ] User runs each available mode's golden behaviour check and representative search/refinement/follow-up checks. Unexpected additional empty answers versus baseline or provider failures block that mode's acceptance.
- [ ] Validate YAML and both effective configurations; score weights sum to 1.0 and changing cloud overrides does not silently alter local settings. Run offline settings/reranking tests and the full backend suite.
- [ ] Document the actual mode, catalog/index, models, dev/test sample sizes, grades, CV and test results, gate and limitations in the new ADR and current docs. A single assessor represents that person's judgments; offline improvements do not prove broad user satisfaction.
- [ ] Commit accepted parameter changes and documentation in separate logical commits. If GPU is unavailable now, retain the unaccepted local state and do not fabricate results; cloud checks are run only by the user.

**Cost and API scope:** Local GPU inference does not incur OpenRouter API charges; electricity and hardware costs still apply. OpenRouter pool export, preflight, interactive checks and an origin-compatibility probe may make paid calls; CPU replay, grading and report do not. An illustrative 2026-10-31 estimate for 20 normal requests returning five films each was about $0.01–0.03 using the then-current Qwen3.5-9B, BGE-M3 and Voyage rerank-3-lite rates; it excludes full-catalog embeddings and is not a quote for the deeper pool export. Check current prices and user-observed usage before projecting a batch: [Qwen](https://openrouter.ai/qwen/qwen3.5-9b), [BGE-M3](https://openrouter.ai/baai/bge-m3), [Voyage](https://openrouter.ai/voyageai/rerank-3-lite).

### Task 8: Remove superseded evaluation only after both modes are accepted

Do not make cleanup a prerequisite for completing implementation or for collecting either mode's evidence. Keep old flags, golden ranking lists and shared signals until graded evaluation is accepted for **both** modes. A zero weight in one mode alone is not grounds to remove a signal used by the other or needed by fitting.

- [ ] Verify accepted, traceable local and OpenRouter run records exist; otherwise mark cleanup deferred.
- [ ] Identify actual callers, then remove only superseded `--sweep`, `--gate-report`, binary label export/report and `--llm-judge` paths and their obsolete tests/settings. Keep shared metric helpers used by graded evaluation.
- [ ] Remove golden expected movie lists and their exclusive resolution code only if no remaining caller needs them. Keep golden queries and behaviour counters, entity probes and report, MMR and the explicit `min_vote_average` filter.
- [ ] Remove a zero-weight signal only when neither accepted profile nor supported fitting depends on it.
- [ ] Update `AGENTS.md`, `docs/ML.md`, `docs/infrastructure.md`, `docs/api.md` and the datasheet. Preserve ADR history by adding supersession notes rather than rewriting old decisions.
- [ ] Add existing-style tests for removed flags; run backend suite offline, validate both profile settings and search for stale references. Commit by logical area.
- [ ] Preserve existing binary labels, user grades, snapshots, manifests and run records. Show an exact list of any other untracked artifacts proposed for removal; delete those only after the user's explicit approval.

## Not in this plan (and why)
- LLM listwise reranking, a bigger or fine-tuned reranker: see the hardware table. Revisit if `junk_at_k` on test stays above the user's bar after Task 7.
- LambdaMART: revisit when there are more than ~2k graded rows and the linear fit has plateaued (CPU-only, no GPU cost).
- An inverted quality signal for niche requests: the user chose "turn quality off".
- Raising `catalog.min_vote_count` at import: it would delete niche films that `wants_obscure` must still reach.
