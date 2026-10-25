# Recommendation Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. First action on execution: copy this file to `docs/superpowers/plans/2026-10-25-recommendation-quality.md` and commit it.

**Goal:** Recommend movies that are both relevant and good, not obscure low-rated filler, except when the user explicitly asks for niche or trashy films. All tuning must be learned from the user's own graded labels.

**Architecture:** Two-stage retrieval → ranking stays as it is. Three changes on top of it, plus one tuning pass:
1. A quality signal (IMDb-style Bayesian weighted rating) enters the scorer.
2. A third retrieval channel puts well-known films into the candidate pool.
3. One LLM-extracted flag (`wants_obscure`) turns both off.

Weights are not hand-picked. They are fitted offline on CPU from graded labels collected with TREC-style pooling. The queries come from Reddit-Movie and are labelled by the user.

**Tech Stack:** Django, Postgres/pgvector, sentence-transformers (bge-m3, bge-reranker-v2-m3), Ollama qwen3:8b, numpy. No new dependencies.

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

**Code:** no quality term in `core/scoring.py` (semantic 0.4667, metadata 0.3333, popularity 0.2). The reranker text in `core/reranking.py:158` is title, genres and overview only. The existing labels are "relevant yes/no", not "would recommend", so they cannot train quality.

**Industry and papers:**
- Two-stage retrieval → ranking is the standard architecture ([YouTube DNN](https://research.google/pubs/pub45530)).
- Production systems blend several retrievers ([Shaped](https://shaped.ai/blog/anatomy-of-modern-ranking-architectures-part-2)).
- Popularity and quality features help learned rankers but can dominate relevance features, so keep relevance as a gate ([arXiv 1903.04263](https://arxiv.org/pdf/1903.04263), [arXiv 2410.05763](https://arxiv.org/pdf/2410.05763)).
- Bayesian weighted rating, `WR = v/(v+m)·R + m/(v+m)·C`, as used for IMDb's Top 250 ([write-up](https://neurobiology.substack.com/p/how-imdb-determines-top-250_27)).
- Depth-k pooling across system variants, shown to assessors in random order with a graded scale ([arXiv 2211.00981](https://arxiv.org/pdf/2211.00981), [Lin et al.](https://arxiv.org/pdf/2010.06467)).
- Retrieval-stage recall is a common bottleneck in reranked movie recommenders ([arXiv 2604.16318](https://arxiv.org/pdf/2604.16318)).
- Extracting a structured preference slot from the query with an LLM is current industry practice ([Instacart](https://tech.instacart.com/building-the-intent-engine-how-instacart-is-revamping-query-understanding-with-llms-3ac8051ae7ac), [AACL 2025](https://underline.io/lecture/138086-hint-augmented-re-ranking-efficient-product-search-using-llm-based-query-decomposition)).
- Query source: [Reddit-Movie](https://huggingface.co/datasets/ZhankuiHe/reddit_movie_large_v1) from [He et al. 2023](https://arxiv.org/pdf/2308.10053), IMDb-linked and research-use only. Fallback: ReDial, CC BY 4.0.

## Hardware trade-off (GTX 1080 Ti, 11 GB, Pascal sm_61)

The VRAM budget is already estimated at 11.3–11.8 GB of 11 GB (`params.yaml` header). Pascal has very slow fp16, so half precision buys nothing.

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
- Never change `embedding.*` or the embedding text: no re-embedding in this plan.
- `wants_obscure: true` turns off the quality signal and the quality channel, nothing else. The quality signal is never inverted.
- The LLM judge is not ground truth (CLAUDE.md). Only the user's grades count.
- Grades: `0` would not recommend, `1` acceptable, `2` good pick, `3` great pick, judged against the query text.
- Docker never runs on the dev Mac. GPU and DB steps are run by the user on the GPU box. Host tests: `cd backend && uv run pytest`.
- Commits: one per logical part, no Claude attribution, never push.
- Nothing obsolete stays: whatever the graded evaluation replaces is deleted in Task 8 (code, tests, params keys, doc instructions, local artifacts with the user's yes). The golden set stays only as a behaviour check.

## Review Focus
1. A movie with `vote_count` 0 or `vote_average` 0.0 (model defaults) gets WR = C, a neutral value: no zero, no crash. Owned by Task 1.
2. Hard filters leave the quality channel's subset empty (for example "1950s Korean documentaries"). Candidate generation falls back to the other channels without error. Owned by Task 2.
3. `wants_obscure` together with `reference_films` or a `min_vote_average` filter: the flag only disables quality, and the explicit rating filter still applies. Owned by Task 3.
4. An empty catalog makes `catalog_mean_rating()` return None. The quality signal collapses to `neutral_score` for the whole pool. Owned by Task 1.
5. A graded CSV with a blank grade, a grade outside 0–3, or a query with fewer than 2 graded rows: rows are skipped and counted, and the error names the line for invalid values. Owned by Task 6.

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

### Task 5: Pooled graded export (one GPU pass)

**Files:**
- Modify: `backend/core/evaluation.py` (new `POOL_EXPORT_COLUMNS`, `pool_export_rows`)
- Modify: `backend/movies/management/commands/evaluate_scoring.py`: new `--queries PATH` (a label-query file instead of the golden set; e2e intent only) and `--pool-export PATH`, mutually exclusive with the other modes, the same way `--label-export` is now
- Modify: `backend/core/pipeline.py`: `generate_and_score(..., rerank_top_k: int | None = None, quality_channel: bool | None = None)`, passed through to `rerank_candidates` and `generate_candidates`; defaults unchanged
- Test: `backend/core/tests/test_evaluation.py`, `backend/movies/tests/test_evaluate_scoring.py`

**Interfaces:**
- Consumes: `generate_candidates(..., quality_channel=True)` (Task 2), `raw_scores` incl. `quality` (Task 1), `intent["wants_obscure"]` (Task 3), and the label-query records (Task 4).
- Produces: one row per (query, pooled movie) with these `POOL_EXPORT_COLUMNS`:
  - query fields: `query_id, query, split, wants_obscure`;
  - movie fields: `tmdb_id, title, year, overview, vote_average, vote_count, popularity`;
  - scores: `rerank_score, semantic_raw, metadata_raw, popularity_raw, quality_raw, lexical_rank, in_quality_channel`;
  - `community_rec` (0/1), `catalog_mean`, `pool_size`, and an empty `grade`.
- Pooling: the union of the top `evaluation.pool_export.depth` (params, 10) under each ranker in `evaluation.pool_export.rankers`. The rankers are: shipped blend; blend with `quality` weight 0.3; pure `rerank_score`; WR among rows with `rerank_score >= min_score`. All `community_rec` films found anywhere in the reranked pool are added too. Rows are shuffled per query with `evaluation.pool_export.seed`, and the source ranker is hidden from the labeller (TREC practice).
- The run summary prints pool recall of `community_recs`: the share found in the candidate pool, with and without the quality channel. This number says whether retrieval is the bottleneck.
- Export reranks with `rerank_top_k=evaluation.pool_export.rerank_top_k` (params, 150). This is offline only: ~2 s per query on the 1080 Ti, about 4 minutes for 120 queries.

- [ ] Step 1: failing tests:
  - `pool_export_rows` dedups the union, adds pool-only community recs, gives the same shuffle for the same seed, and leaves `grade` empty;
  - argument validation rejects `--pool-export` with `--sweep`, and `--queries` with `--intent oracle`.
- [ ] Step 2: FAIL. Step 3: implement. Step 4: PASS plus the full host suite.
- [ ] Step 5: commit `feat: export a pooled, feature-rich candidate set for graded labelling`.
- [ ] Step 6 (user, GPU box): `docker compose exec backend python manage.py evaluate_scoring --queries data/label_queries.json --pool-export eval_runs/pool.csv`, then grade every row 0–3 for the dev split first (roughly 120 × 20 rows; dev about 80 queries).

### Task 6: Offline grade report and weight fitting (CPU, no DB)

**Files:**
- Modify: `backend/core/evaluation.py`: `grade_report`, `replay_ranking`, `fit_ranking_params`
- Modify: `backend/movies/management/commands/evaluate_scoring.py`: `--grade-report PATH [--fit]`, needing no DB or model, like `--label-report`
- Modify: `params.yaml` new `evaluation.grade_fit` block: `folds: 5`, `top_k: 5`, `grid` (candidate values for each weight, `reranking.weight`, `scoring.quality.prior_votes`), `seed: 7`
- Test: `backend/core/tests/test_evaluation.py`, `backend/movies/tests/test_evaluate_scoring.py`

**Interfaces:**
- Consumes: Task 5's CSV.
- Produces: `replay_ranking(rows_for_query: list[dict], params: dict) -> list[dict]`. It recomputes WR from `vote_average`, `vote_count`, `catalog_mean` and `prior_votes`, min-max normalizes each signal over the query's rows the way `score_candidates` does, blends with min-max `rerank_score` at `reranking.weight`, applies `min_score`, and sorts. The orientation matches production, so fitted numbers transfer.
- Produces: `grade_report(rows, params, k) -> dict` with:
  - `ndcg_at_k` (graded gains `2^g - 1`);
  - `junk_at_k` (share of grade 0 in the top-k);
  - `good_at_k` (share ≥2);
  - `community_recall_at_k`;
  - all of the above split by `wants_obscure`;
  - skipped-row counts;
  - a gate table for `reranking.min_score`. For each threshold in `evaluation.grade_fit.gate_thresholds` (params, the same values as today's `GATE_REPORT_THRESHOLDS`) it reports kept share ≥1, relevant (grade ≥1) dropped, and queries emptied. This replaces `label_report`, which Task 8 deletes.
- Produces: `fit_ranking_params(rows, grid, folds, seed, k) -> dict`. This is a grouped K-fold over query_id. Inner selection is an exhaustive grid when the grid has ≤ 5k combinations, otherwise coordinate ascent; weights are constrained to sum to 1. It returns the best params, the CV mean and standard deviation of `ndcg_at_k` and `junk_at_k`, the shipped-params baseline on the same folds, and a ready-to-paste `params.yaml` snippet.

- [ ] Step 1: failing tests:
  - `replay_ranking` with shipped params reproduces the export order on a fixture where the export order is known;
  - `grade_report` on a hand-computed 2-query fixture gives exact nDCG@2 and junk@2;
  - invalid grade "4" raises an error naming the CSV line, and a blank grade is skipped and counted;
  - `fit_ranking_params` on a synthetic set where the grade is monotonic in `quality_raw` picks quality weight > 0, and no fold contains a query from another fold;
  - `test_gate_table_from_grades`: on a 2-query fixture, threshold 0.5 gives the hand-computed dropped and emptied shares.
- [ ] Step 2: FAIL. Step 3: implement with numpy only. Step 4: PASS.
- [ ] Step 5: commit `feat: report and fit ranking weights from graded labels offline`.

### Task 7: Ship the fitted values and record the decision

**Files:**
- Modify: `params.yaml`: fitted `scoring.weights.*`, `scoring.quality.prior_votes`, `retrieval.rrf_weights.quality`, `reranking.weight`, and `reranking.min_score` recalibrated from the grade gate table with the same rule as before (0% answerable queries emptied, ≤ ~5% grade ≥1 dropped), each with a comment naming the fit run (date, number of queries and grades, CV nDCG@5 and junk@5 against baseline)
- Create: `docs/adr/0012-quality-prior-and-learned-weights.md` (context = the evidence above, decision, hardware trade-off table, consequences incl. "niche only via `wants_obscure`"); row in `docs/adr/README.md`
- Modify: `docs/ML.md` (Hybrid Scoring table gets `quality`; new retrieval channel; Evaluation Results gets the fit numbers), `docs/core.md` pipeline diagram
- Test: `backend/core/tests/test_reranking.py` / settings tests pin that shipped weights load and sum to 1

- [ ] Step 1 (user): `cd backend && python manage.py evaluate_scoring --grade-report ../eval_runs/pool.csv --fit` on dev grades, then paste the snippet.
- [ ] Step 2: apply the snippet and write the docs; host tests PASS.
- [ ] Step 3 (user, GPU box): grade the test-split rows, then run `--grade-report` with shipped params on test only. Ship if `junk_at_k` drops and `ndcg_at_k` does not drop against baseline on test. Rerun `evaluate_scoring --split all` and the classify `prompt_tokens` grep; `empty_results` must stay 0.
- [ ] Step 4: commit `feat: ship quality weights fitted on graded labels` and `docs: record the quality prior in ADR 0012`.

### Task 8: Delete what the new evaluation replaces

Runs after Task 7, once the grade-based fit and gate are shipped. Golden set: keep it for behaviour (the user's choice). Delete its movie lists and the ranking metrics computed from them.

**Delete:**
- **Signals.** Any scorer signal whose fitted weight in Task 7 is 0, `popularity` included if so: the code in `core/scoring.py`, its `params.yaml` keys, its tests and docs. No dead zero-weight signal stays.
- **`evaluate_scoring` modes and their code:**
  - `--sweep` (`_run_sweep`, golden-list weight grid; replaced by `--grade-report --fit`);
  - `--gate-report` (`gate_report`, `_write_gate_report`, `GATE_REPORT_*`);
  - `--label-export` / `--label-report` (`label_export_rows`, `LABEL_EXPORT_COLUMNS`, `label_report`, `_write_label_export`, `_read_labelled_csv`, `_write_label_report`);
  - `--llm-judge` (`judge_ndcg`, the judge call path, `params.yaml` `evaluation.judge_*`; circular by CLAUDE.md's own rule);
  - `--k` if nothing else uses it.
  - Delete their tests in `backend/core/tests/test_evaluation.py` and `backend/movies/tests/test_evaluate_scoring.py`. Keep `test_session_ablation_flag_is_gone`-style "flag is gone" tests only where the repo already uses that pattern, and add one per removed flag in the same style.
- **Golden set ranking parts:**
  - `precision_at_k`, `recall_at_k`, `ndcg_at_k` over golden lists in `core/evaluation.py`;
  - the per-entry expected movie lists in `data/golden_set.json`, plus the reference resolution in `core/golden_set.py` that only serves them (`resolved_refs`, `unresolved_refs`, `unscorable`), if nothing else uses it.
  - `evaluate_scoring --split ...` still reports `misclassified`, `expected_empty(_correct)`, `expected_ask(_correct)`, `empty_results`, `possible_parse_fallbacks` and `unsupported_constraints`.
  - Update `docs/datasheet.md` and the golden-set note in `docs/ML.md` to say what the set is for now.
- **Docs and commands:**
  - CLAUDE.md "Commands": drop the sweep, gate-report, label-export and label-report lines and the stale `--session-ablation` line (the flag no longer exists); add `--pool-export` and `--grade-report --fit`. Drop the `--llm-judge` anti-pattern line once the flag is gone.
  - `docs/ML.md`, `docs/infrastructure.md`, `docs/api.md`, ADR 0003/0007/0009: replace the instructions for removed modes. ADRs keep their history: add a "Superseded by ADR 0012" note instead of rewriting past decisions.
- **Local artifacts** (user-approved, untracked): `eval_runs/labels.csv`, `before.json`, `after.json`, `after2.json`, `after3.json`, `eval-*.json`, `gate_report-*.json`, `label_export-*.json`, `session_ablation-*.json`, `entity_resolution_report.txt`, `pytest.txt`. Show the exact list to the user and delete only after a yes. Keep `eval_runs/pool.csv` (the graded labels) and the run records the new modes write.

**Keep:** `data/entity_probes.json` and `entity_resolution_report` (entity resolution is untouched), the `min_vote_average` hard filter, MMR, and the golden set queries and behaviour counters.

- [ ] Step 1: grep proves each deleted symbol has no remaining caller (`grep -rn "<symbol>" backend docs CLAUDE.md params.yaml`). Paste the empty results into the report.
- [ ] Step 2: delete code, tests, params keys and doc lines; add the "flag is gone" tests; host suite PASS with no new failures; `python -c "import yaml; yaml.safe_load(open('params.yaml'))"` loads and settings validation accepts it.
- [ ] Step 3: commits, one per area: `refactor: remove the golden-list ranking metrics`, `refactor: remove the binary-label and golden-hit gate reports`, `refactor: remove the LLM judge`, `refactor: remove zero-weight scoring signals`, `docs: point commands and ADRs at the graded evaluation`.
- [ ] Step 4 (user-confirmed): delete the listed `eval_runs/` files.

## Not in this plan (and why)
- LLM listwise reranking, a bigger or fine-tuned reranker: see the hardware table. Revisit if `junk_at_k` on test stays above the user's bar after Task 7.
- LambdaMART: revisit when there are more than ~2k graded rows and the linear fit has plateaued (CPU-only, no GPU cost).
- An inverted quality signal for niche requests: the user chose "turn quality off".
- Raising `catalog.min_vote_count` at import: it would delete niche films that `wants_obscure` must still reach.
