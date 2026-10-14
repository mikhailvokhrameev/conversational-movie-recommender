# 0009. The relevance gate ships off until it is calibrated

Status: Proposed (built and tested; the threshold is not chosen)

## Context

Retrieval always has a top 5, so it can never answer "the catalog has nothing for
this". Semantic cosine after min-max, genre overlap, popularity and RRF rank are all
relative to the rest of the pool. Only the cross-encoder's raw score is an absolute
answer to "does this movie match this query". A threshold on it could let the system
say "nothing matched" instead of padding. But a threshold nobody measured is the defect
already found in `intent.genre_match_threshold` (an eyeballed 0.5): too high silently
empties answerable queries, too low gates nothing.

## Decision

Build the gate, ship it disabled. `reranking.min_score: null` means no gate. When set,
`core.pipeline.apply_relevance_gate` keeps candidates with `rerank_score >= min_score`
before MMR, and drops candidates the reranker never scored. Both `ChatView` and
`evaluate_scoring` run it through `generate_and_score`. An empty result makes the chat
reply with a fixed "nothing matched" message, skips the explanation LLM call and
reports `no_match: true`. It fails open: if the reranker is disabled, failed or did not
score the request, the gate is skipped with a warning, because an infrastructure fault
must not become zero results. `min_score` must lie strictly inside (0, 1) (the score is
a sigmoid output) or be null; settings refuse anything else.

The value is to be chosen by hand from `evaluate_scoring --intent oracle --gate-report`
on the dev split on the GPU box, then judged on the test split. The command never writes
`params.yaml`.

## Consequences

- Behaviour is unchanged until someone calibrates; no number is shipped on a guess.
- Calibration is blocked on the golden set review and an imported catalog
  ([0007](0007-evaluation-methodology.md)).
- The golden set has no unanswerable queries, so the report shows the cost of a
  threshold (hits dropped, queries emptied) but not its benefit. Queries the catalog
  cannot serve must be added before trusting a chosen value.
- The [0..1] range rests on reading sentence-transformers 2.7.0 source; the model was
  not available on the build host, so the report prints the observed min/max to confirm.
- This record moves to Accepted once a value is set and checked on test.

Details: [ML.md, "Nothing suitable" gate](../ML.md#cross-encoder-reranking),
[core.md, pipeline.py](../core.md#modules), `reranking.min_score` in
[params.yaml](../../params.yaml), `TODOS.md` "Relevance gate".
