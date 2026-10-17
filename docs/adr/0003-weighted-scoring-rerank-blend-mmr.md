# 0003. Min-max weighted scoring, cross-encoder blend, MMR

Status: Accepted

## Context

The ~100 fused candidates must become five results. Signals have very different
spreads: semantic and session cosines sit in a narrow band, popularity is unbounded and
right-skewed, genre overlap spans 0-1 in jumps. A bi-encoder compares query and movie
without seeing both together. A top-5 by relevance alone tends to return one franchise.

## Decision

Three stages, in this order, all in `core/`:

1. **Weighted sum of min-max normalized signals** (`scoring.score_candidates`):
   semantic 0.35, metadata 0.25, session 0.25, popularity 0.15. Normalizing each signal
   across the candidate set first makes a weight equal its actual influence
   (`weight * spread`). A signal identical for every candidate collapses to a neutral 0.5.
   Popularity is a signal, not a filter, because TMDB's metric is unbounded.
2. **Cross-encoder rerank** (`bge-reranker-v2-m3`, top 50), blended 50/50 with the
   scorer total after min-max on both sides (`reranking.weight`). Blend, not replace:
   the cross-encoder knows nothing about the session or the parsed genre intent.
3. **MMR** (`lambda` 0.7, `diversification.top_n` 5) using embedding similarity, which
   catches same-franchise and same-sub-genre redundancy that "max 2 per genre" rules
   cannot.

RRF in [0002](0002-hybrid-retrieval-exact-knn-lexical-rrf.md) and min-max here are
deliberately different: retrieval fuses incomparable scores from independent systems
(rank only); scoring ranks one shared pool whose magnitudes are real information.

## Consequences

- Weights are explicit, validated at load (non-negative, sum to 1.0) and tunable with
  `evaluate_scoring --sweep` on the dev split.
- `reranking.top_k` is the latency dial (50 pairs ~0.64 s measured on the GTX 1080 Ti).
  The reranker needs the GPU; on CPU it takes seconds per request. It fails safe to the
  scorer order if it cannot load.
- Whether reranking helps here is not measured yet; `reranking.weight`, `top_k` and the
  scoring weights are untuned defaults until a reviewed golden set is run
  ([0007](0007-evaluation-methodology.md)).
- Weights from before the catalog swap do not apply (different catalog, embedder and
  signal set).

Details: [ML.md, Hybrid Scoring](../ML.md#hybrid-scoring),
[Cross-Encoder Reranking](../ML.md#cross-encoder-reranking),
[MMR Diversification](../ML.md#mmr-diversification).
