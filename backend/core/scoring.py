from django.conf import settings

from core.embedding_service import cosine_similarity

SIGNALS = ("semantic", "metadata", "session")

# scoring.neutral_score and diversification.* come from params.yaml.
# Avoiding division by a near-zero range
_MIN_SPREAD = 1e-9


def score_candidates(
    candidates: list[dict],
    query_embedding: list[float],
    intent: dict,
    session_vector: list[float] | None = None,
    weights: dict | None = None,
) -> list[dict]:
    """Score every candidate, normalizing each signal across the whole set.

    Scoring is set-level rather than per-movie because normalization needs
    the full candidate pool to know each signal's actual range.

    Returns new dicts (candidates are not mutated), each carrying the
    normalized per-signal scores, the weighted `total`, and the pre-
    normalization values under `raw_scores` for debugging and evaluation.
    """
    if not candidates:
        return []

    w = weights or settings.SCORE_WEIGHTS

    raw = {
        "semantic": [_semantic_score(c, query_embedding) for c in candidates],
        "metadata": [_metadata_score(c, intent) for c in candidates],
        "session": [
            _session_score(c, session_vector) if session_vector else 0.0
            for c in candidates
        ],
    }
    normalized = {signal: normalize_scores(values) for signal, values in raw.items()}

    scored = []
    for i, candidate in enumerate(candidates):
        signal_scores = {signal: normalized[signal][i] for signal in SIGNALS}
        scored.append(
            {
                **candidate,
                **signal_scores,
                "total": sum(signal_scores[s] * w[s] for s in SIGNALS),
                "raw_scores": {signal: raw[signal][i] for signal in SIGNALS},
            }
        )
    return scored


def mmr_diversify(
    scored_candidates: list[dict],
    top_n: int | None = None,
    lambda_param: float | None = None,
) -> list[dict]:
    """Select top-N diverse results via Maximal Marginal Relevance.

    Each candidate dict must have 'total' (relevance score) and
    'embedding' (for pairwise similarity computation).

    Note: `max_sim` here is a raw cosine in [-1, 1] while `relevance` is in
    [0, 1], so the diversity penalty is on a different scale than relevance.
    Left as-is deliberately: correcting it changes ranking behaviour, and
    there is no golden-set evaluation yet to measure whether the change
    helps. Revisit once retrieval metrics exist.
    """
    top_n = top_n if top_n is not None else settings.TOP_N
    lambda_param = (
        lambda_param if lambda_param is not None else settings.MMR_LAMBDA
    )

    if len(scored_candidates) <= top_n:
        return scored_candidates

    candidates = list(scored_candidates)
    selected = []

    best = max(candidates, key=lambda c: c["total"])
    selected.append(best)
    candidates.remove(best)

    while len(selected) < top_n and candidates:
        best_mmr_score = -float("inf")
        best_candidate = None

        for candidate in candidates:
            relevance = candidate["total"]

            max_sim = 0.0
            cand_emb = candidate.get("embedding")
            if cand_emb:
                for sel in selected:
                    sel_emb = sel.get("embedding")
                    if sel_emb:
                        sim = cosine_similarity(cand_emb, sel_emb)
                        max_sim = max(max_sim, sim)

            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr > best_mmr_score:
                best_mmr_score = mmr
                best_candidate = candidate

        if best_candidate:
            selected.append(best_candidate)
            candidates.remove(best_candidate)
        else:
            break

    return selected


def normalize_scores(values: list[float]) -> list[float]:
    """Min-max normalize a signal's values to [0, 1] across the candidate set.

    A signal where every candidate scores the same carries no ranking
    information, so it collapses to NEUTRAL_SCORE instead of being stretched
    across the full range by numerical noise.
    """
    if not values:
        return []

    low, high = min(values), max(values)
    spread = high - low
    if spread < _MIN_SPREAD:
        return [settings.NEUTRAL_SCORE] * len(values)

    return [(value - low) / spread for value in values]


def _embedding_similarity(vec: list[float], movie: dict) -> float:
    """Cosine similarity between a vector and the movie's embedding, mapped to [0, 1]."""
    movie_embedding = movie.get("embedding")
    if movie_embedding is None:
        return 0.0
    sim = cosine_similarity(vec, movie_embedding)
    return max(0.0, (sim + 1.0) / 2.0)


def _semantic_score(movie: dict, query_embedding: list[float]) -> float:
    return _embedding_similarity(query_embedding, movie)


def _metadata_score(movie: dict, intent: dict) -> float:
    """Genre overlap between LLM-extracted intent genres and movie genres."""
    intent_genres = set(intent.get("genres", []))
    movie_genres = set(movie.get("genres", []))

    if not intent_genres:
        return 0.5

    overlap = len(intent_genres & movie_genres)
    return overlap / len(intent_genres)


def _session_score(movie: dict, session_vector: list[float]) -> float:
    return _embedding_similarity(session_vector, movie)
