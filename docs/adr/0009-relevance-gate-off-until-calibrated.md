# 0009. The relevance gate ships off until it is calibrated

Status: Proposed (built, tested and calibrated on dev: `min_score: 0.005`; the test-split check is pending)

## Context

Retrieval always has a top 5, so it can never answer "the catalog has nothing for
this". Semantic cosine after min-max, genre overlap, popularity and RRF rank are all
relative to the rest of the pool. Only the cross-encoder's raw score is an absolute
answer to "does this movie match this query". A threshold on it could let the system
say "nothing matched" instead of padding. But a threshold nobody measured is the defect
already found in `intent.genre_match_threshold` (an eyeballed 0.5): too high silently
empties answerable queries, too low gates nothing.

## Decision

Build the gate, ship it disabled until calibrated (now set to 0.005, see the calibration
result below). `reranking.min_score: null` means no gate. When set,
`core.pipeline.apply_relevance_gate` keeps candidates with `rerank_score >= min_score`
before MMR, and drops candidates the reranker never scored. Both `ChatView` and
`evaluate_scoring` run it through `generate_and_score`. An empty result makes the chat
reply with a fixed "nothing matched" message, skips the explanation LLM call and
reports `no_match: true`. It fails open: if the reranker is disabled, failed or did not
score the request, the gate is skipped with a warning, because an infrastructure fault
must not become zero results. `min_score` must lie strictly inside (0, 1) (the score is
a sigmoid output) or be null; settings refuse anything else.

The value is to be chosen by hand on the dev split on the GPU box, then judged on the
test split. Procedure: export with `evaluate_scoring --intent oracle --k 10
--label-export eval_runs/labels.csv` (the container's writable mount, `./eval_runs` on the
host), mark each row's `label` 1/0 by hand (the golden lists are
incomplete, so `--gate-report`'s "hit" undercounts relevance), run `evaluate_scoring
--label-report ../eval_runs/labels.csv` on the host from `backend/` for the absolute and relative curves, pick, then confirm on
test. The label report splits emptied queries the same way as `--gate-report`: a query
with a relevant label that empties is a cost, a query labelled all 0 that empties is the
benefit; an absolute threshold empties a query when its pool's best score
(`pool_max_rerank_score`) is below it, while kept precision and relevant dropped are
measured over the exported top-k, a sample of what the gate filters. `--gate-report` remains as the golden-hit view; next to the cost columns it
prints the benefit, `unanswerable_emptied_share` (the share of the golden set's
`expected: "empty"` queries a threshold would correctly empty), and the label export
includes those queries' rows as negatives to mark 0. The command never writes
`params.yaml`.

## Consequences

- The gate shipped off until it was calibrated; the value now set (0.005) comes from
  hand labels, not from a guess (calibration result below).
- Calibration needed the golden set and an imported catalog
  ([0007](0007-evaluation-methodology.md)).
- The golden set now holds `expected: "empty"` queries (five draft entries, two on
  dev), so the report prices a threshold's benefit as well as its cost. Two dev
  entries cannot pick a value on their own; more unanswerable queries, reviewed like the
  rest, are needed before a chosen value is trusted.
- The [0..1] range rests on reading sentence-transformers 2.7.0 source; the model was
  not available on the build host, so the report prints the observed min/max to confirm.
- 2026-10-17 calibration run (oracle, dev, GPU): 6 hits against 129 non-hits, scores
  inside [0, 1]. A threshold of 0.01 drops no hits and 14% of non-hits; 0.02 empties
  7.4% of queries. Six hits were too few to choose a value, so the gate stayed off
  (superseded by the 2026-10-25 hand-label calibration below).
- This record moves to Accepted once a value is set and checked on test (see the
  calibration result below).

## Calibration result (2026-10-25)

Hand labels on the dev split: 30 queries x top-10 = 300 rows, 139 relevant; 27
answerable queries and 3 whose every row is irrelevant. Read with
`evaluate_scoring --label-report`. Absolute thresholds (kept precision is over the
exported top-10; the baseline with no gate is 46.3%):

| min_score | kept precision | relevant dropped | answerable queries emptied | unanswerable emptied |
|---|---|---|---|---|
| 0.001 | 46.3% | 0.0% | 0.0% | 0.0% |
| **0.005** | **47.7%** | **4.3%** | **0.0%** | **0.0%** |
| 0.010 | 47.0% | 10.1% | 0.0% | 0.0% |
| 0.020 | 49.0% | 13.7% | 0.0% | 0.0% |
| 0.050 | 51.0% | 25.9% | 7.4% | 0.0% |
| 0.100 | 53.1% | 31.7% | 14.8% | 0.0% |
| 0.200 | 55.7% | 40.3% | 29.6% | 66.7% |

The relative variant (keep a movie within a margin of the query's best score) lost
at every margin from 0.01 to 0.70: kept precision 35.7-47.7%, never more than 1.4
points above the 46.3% baseline, relevant movies dropped 1.4-89.9%. The absolute
threshold stays.

Chosen value: `reranking.min_score: 0.005`. Rule: no answerable query emptied and at
most ~5% of the relevant movies dropped; 0.005 is the last row that holds it (0.010
already drops 10.1%, 0.05 empties 7.4% of answerable queries).

Consequence: the gate trims junk; it is not an "empty" detector. The best pool scores
on unanswerable queries reach 0.65 while some answerable queries peak at only
0.025-0.15, so no absolute threshold separates the two, and the rows that do empty the
unanswerable queries (0.20 and up) also empty a third of the answerable ones.
`expected_empty_correct` stays 0/5 on the golden set because the unanswerable pools
score 0.027-0.65. Saying "nothing in the catalog" needs a different signal.

Moves to Accepted once `evaluate_scoring --split test --intent oracle` reports
`counters.empty_results == 0` with this value.

Details: [ML.md, "Nothing suitable" gate](../ML.md#cross-encoder-reranking),
[core.md, pipeline.py](../core.md#modules), `reranking.min_score` in
[params.yaml](../../params.yaml), `TODOS.md` "Relevance gate".
