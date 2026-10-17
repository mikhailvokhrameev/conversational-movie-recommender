"""The retrieval pipeline shared by the chat view and offline evaluation.

generate_candidates -> score_candidates -> rerank_candidates -> mmr_diversify.

It lives here, as one plain sync function, so `ChatView` (via a thin
`sync_to_async` wrapper) and `manage.py evaluate_scoring` run literally the
same code: an evaluation that re-implements the pipeline measures its own copy,
and the two drift apart the first time either side changes.
"""

import logging

from django.conf import settings

from core.candidate_generation import generate_candidates
from core.reranking import rerank_candidates
from core.scoring import mmr_diversify, score_candidates

logger = logging.getLogger(__name__)


def apply_relevance_gate(candidates: list[dict], min_score: float | None) -> list[dict]:
    """The "nothing suitable" gate: keep candidates with rerank_score >= min_score.

    Only the cross-encoder's absolute score can say "this does not match" --
    every other signal is relative to the rest of the pool, so top-5 always
    exists. Candidates without a rerank_score ranked below the reranked head
    and are dropped with the failures.

    Fails open: when no candidate carries a rerank_score the reranker was
    disabled, unavailable, or failed on this request, and gating on a score that
    does not exist would turn an infrastructure fault into zero results. The
    pool is returned as-is and that is logged once.
    """
    if min_score is None or not candidates:
        return candidates
    if not any(c.get("rerank_score") is not None for c in candidates):
        logger.warning(
            "reranking.min_score is set but the reranker did not score this request "
            "(disabled, unavailable, or failed): relevance gate skipped"
        )
        return candidates
    return [
        c for c in candidates
        if c.get("rerank_score") is not None and c["rerank_score"] >= min_score
    ]


def generate_and_score(
    query_embedding: list[float],
    intent: dict,
    session_vector: list[float] | None,
    query_text: str = "",
    weights: dict | None = None,
    top_n: int | None = None,
    apply_gate: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Returns (top_movies, candidate_pool): candidate_pool is the full reranked
    pool before the top-n cut, kept around cheaply for the hallucination check
    in the explanation stream's caller -- no extra retrieval work, no DB round trip.

    `weights` and `top_n` default to the params.yaml values; only evaluation
    overrides them (weight sweeps, a custom k). `apply_gate=False` skips the
    reranking.min_score gate -- only the gate calibration report uses it, to see
    the scores of results the gate would have removed. With the gate on, the
    top list can be shorter than `top_n`, or empty; `candidate_pool` is still
    the full reranked pool.
    """
    candidates = generate_candidates(query_embedding, intent, query_text=query_text)
    scored = score_candidates(candidates, query_embedding, intent, session_vector, weights=weights)
    reranked = rerank_candidates(query_text, scored)
    gated = apply_relevance_gate(reranked, settings.RERANK_MIN_SCORE) if apply_gate else reranked
    top_movies = mmr_diversify(gated, top_n=top_n if top_n is not None else settings.TOP_N)
    return top_movies, reranked
