"""The retrieval pipeline shared by the chat view and offline evaluation.

generate_candidates -> score_candidates -> rerank_candidates -> mmr_diversify.

It lives here, as one plain sync function, so `ChatView` (via a thin
`sync_to_async` wrapper) and `manage.py evaluate_scoring` run literally the
same code: an evaluation that re-implements the pipeline measures its own copy,
and the two drift apart the first time either side changes.
"""

from django.conf import settings

from core.candidate_generation import generate_candidates
from core.reranking import rerank_candidates
from core.scoring import mmr_diversify, score_candidates


def generate_and_score(
    query_embedding: list[float],
    intent: dict,
    session_vector: list[float] | None,
    query_text: str = "",
    weights: dict | None = None,
    top_n: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Returns (top_movies, candidate_pool): candidate_pool is the full reranked
    pool before the top-n cut, kept around cheaply for the hallucination check
    in the explanation stream's caller -- no extra retrieval work, no DB round trip.

    `weights` and `top_n` default to the params.yaml values; only evaluation
    overrides them (weight sweeps, a custom k).
    """
    candidates = generate_candidates(query_embedding, intent, query_text=query_text)
    scored = score_candidates(candidates, query_embedding, intent, session_vector, weights=weights)
    reranked = rerank_candidates(query_text, scored)
    top_movies = mmr_diversify(reranked, top_n=top_n if top_n is not None else settings.TOP_N)
    return top_movies, reranked
