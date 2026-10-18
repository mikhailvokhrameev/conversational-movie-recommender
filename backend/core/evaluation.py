"""Offline evaluation metrics for recommendation quality.

Three categories of metrics:
  Accuracy:        precision@k, recall@k, ndcg@k against golden-set labels
  Beyond-accuracy: diversity, novelty, coverage
  LLM-as-judge:    graded relevance scoring via Ollama (opt-in, slow)

plus `gate_report`, the calibration aid for the reranking.min_score relevance
gate (it informs a human choice of threshold; it measures no ranking quality).

Relevance is membership in the golden set's resolved movie ids, with optional
grades -- never genre overlap: the `metadata` scoring signal *is* genre
overlap with the intent, so a genre-defined "hit" would score the ranker
against its own input.

Usage:
  from core.evaluation import evaluate_query, aggregate_metrics
  metrics = evaluate_query(recommended_ids, {movie_id: grade}, embeddings, k=5)
"""

import json
import logging
import math
from collections import Counter
from collections.abc import Hashable

import httpx
import numpy as np
from django.conf import settings

from core.embedding_service import cosine_similarity
from core.ollama_client import _chat_sync, _parse_json_response

logger = logging.getLogger(__name__)

# The judge's scale, fixed by LLM_JUDGE_PROMPT below -- not a tuning knob.
JUDGE_MIN_SCORE = 1
JUDGE_MAX_SCORE = 5

# Sent as Ollama's `format`, so decoding can only produce an integer on the
# scale. llm_relevance_score still range-checks the parsed value.
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": JUDGE_MIN_SCORE, "maximum": JUDGE_MAX_SCORE},
    },
    "required": ["score"],
}

LLM_JUDGE_PROMPT = """Rate how relevant this movie is to the user's query.

User query: "{query}"
Movie: "{title}" ({genres})
Description: {description}

Rate relevance from 1 to 5:
5 = perfect match for what the user wants
4 = good match, relevant genre and theme
3 = acceptable, partially relevant
2 = weak match, only tangentially related
1 = irrelevant to the query

Return ONLY a JSON object: {{"score": N}}"""


def precision_at_k(recommended: list[Hashable], relevant: list[Hashable], k: int = 5) -> float:
    if k <= 0:
        return 0.0
    top_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / k


def recall_at_k(recommended: list[Hashable], relevant: list[Hashable], k: int = 5) -> float:
    if not relevant:
        return 0.0
    top_k = set(recommended[:k])
    relevant_set = set(relevant)
    hits = len(top_k & relevant_set)
    return hits / len(relevant_set)


def _dcg(gains: list[float]) -> float:
    return sum(gain / math.log2(i + 2) for i, gain in enumerate(gains))


def ndcg_at_k(
    recommended: list[Hashable],
    grades: dict[Hashable, float],
    k: int = 5,
) -> float:
    """nDCG@k with graded golden labels.

    DCG uses each returned item's golden grade (0 when it is not in the golden
    set); IDCG is the golden grades themselves sorted descending, top-k. The
    ideal comes from the labels, never from the returned list -- otherwise a
    list of uniformly bad results would be its own ideal and score 1.0.
    """
    if k <= 0:
        return 0.0
    dcg = _dcg([grades.get(item, 0.0) for item in recommended[:k]])
    idcg = _dcg(sorted(grades.values(), reverse=True)[:k])
    return dcg / idcg if idcg > 0 else 0.0


def judge_ndcg(scores: list[float], k: int = 5) -> float:
    """nDCG@k over LLM-judge scores (1-5), with gain = score - 1.

    The ideal is k positions at the maximum judge grade, so a list the judge
    rates all 1/5 scores 0.0 and a short list cannot reach 1.0. Expects a fully
    judged list: callers must not fill failed judge calls with a guess.
    """
    if k <= 0:
        return 0.0
    dcg = _dcg([score - JUDGE_MIN_SCORE for score in scores[:k]])
    idcg = _dcg([JUDGE_MAX_SCORE - JUDGE_MIN_SCORE] * k)
    return dcg / idcg


def coverage(all_recommendations: list[list[Hashable]], catalog_size: int) -> float:
    if catalog_size == 0:
        return 0.0
    unique_items = set()
    for rec_list in all_recommendations:
        unique_items.update(rec_list)
    return len(unique_items) / catalog_size


def diversity(embeddings: list[list[float]]) -> float:
    """1 - mean pairwise cosine similarity among recommended items."""
    n = len(embeddings)
    if n < 2:
        return 1.0
    total_sim = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_sim += cosine_similarity(embeddings[i], embeddings[j])
            pairs += 1
    return 1.0 - total_sim / pairs


def novelty(recommended_genres: list[list[str]], genre_freq: dict[str, float]) -> float:
    """How non-obvious the recommendations are, based on genre rarity.

    Uses inverse frequency: rare genres (Documentary) score higher than common ones (Drama).
    Returns value in [0, 1] where 1 = all rare genres, 0 = all common genres.
    """
    if not recommended_genres or not genre_freq:
        return 0.0

    scores = []
    for genres in recommended_genres:
        for g in genres:
            freq = genre_freq.get(g, 0.0)
            scores.append(1.0 - freq)

    return float(np.mean(scores)) if scores else 0.0


def llm_relevance_score(query: str, movie: dict) -> int | None:
    """Ask Ollama to rate relevance of a movie to a query on a 1-5 scale.

    Returns None when the call fails or the reply is not a score on the
    scale. There is no substitute score: a made-up 3 would be averaged in as
    if the judge had said it, so callers skip None and count the failure.
    """
    prompt = LLM_JUDGE_PROMPT.format(
        query=query,
        title=movie.get("serial_name", ""),
        genres=", ".join(movie.get("genres", [])),
        description=(movie.get("description", "") or "")[:settings.EVAL_JUDGE_DESCRIPTION_CHARS],
    )
    try:
        content = _chat_sync(
            [{"role": "user", "content": prompt}],
            call_type="judge",
            schema=JUDGE_SCHEMA,
            timeout=settings.EVAL_JUDGE_TIMEOUT_SECONDS,
        )
        score = _parse_json_response(content)["score"]
        if isinstance(score, bool) or not isinstance(score, int):
            raise ValueError(f"score is not an integer: {score!r}")
        if not JUDGE_MIN_SCORE <= score <= JUDGE_MAX_SCORE:
            raise ValueError(f"score {score} is outside {JUDGE_MIN_SCORE}-{JUDGE_MAX_SCORE}")
        return score
    except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        logger.warning(f"LLM judge failed for '{movie.get('serial_name', '')}': {e}")
        return None


# Candidate thresholds for the gate calibration report. rerank_score is a
# sigmoid output in [0, 1] and bge-reranker scores are heavily skewed toward
# 0, so the low end is finer. A report grid, not a tuning knob.
GATE_REPORT_THRESHOLDS = (
    0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
)


def _distribution(values: list[float]) -> dict:
    keys = ("min", "q1", "median", "q3", "max")
    if not values:
        return {"count": 0, **dict.fromkeys(keys)}
    stats = np.percentile(values, [0, 25, 50, 75, 100])
    return {"count": len(values), **{key: round(float(v), 6) for key, v in zip(keys, stats)}}


def _share(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def gate_report(queries: list[dict], thresholds=GATE_REPORT_THRESHOLDS) -> dict:
    """What a candidate `reranking.min_score` would do, from an ungated run.

    `queries` are per-query results of a run with the gate off, each with
    `unscorable`, `pool_max_rerank_score`, and `recommended` entries carrying
    `rerank_score` and `hit` (membership in the golden labels). Unscorable
    queries have no ground truth to call anything a hit or non-hit, and queries
    with no rerank scores (reranker unavailable) have nothing to threshold, so
    both are left out and counted.

    For each threshold: the share of returned hits and non-hits scoring below
    it (the gate keeps `>=`), and the share of queries that would become empty.
    Emptiness uses the best score in the whole reranked pool, not the returned
    list: the gate runs before MMR on the full pool, so a query empties exactly
    when nothing in the pool reaches the threshold. A share is None when its
    denominator is zero.
    """
    considered = [
        q for q in queries
        if not q["unscorable"] and q.get("pool_max_rerank_score") is not None
    ]
    hit_scores, non_hit_scores = [], []
    for q in considered:
        for movie in q["recommended"]:
            score = movie.get("rerank_score")
            if score is not None:
                (hit_scores if movie["hit"] else non_hit_scores).append(score)

    rows = [
        {
            "min_score": threshold,
            "hits_dropped_share": _share(sum(s < threshold for s in hit_scores), len(hit_scores)),
            "non_hits_dropped_share": _share(
                sum(s < threshold for s in non_hit_scores), len(non_hit_scores),
            ),
            "queries_emptied_share": _share(
                sum(q["pool_max_rerank_score"] < threshold for q in considered), len(considered),
            ),
        }
        for threshold in thresholds
    ]
    return {
        "queries_considered": len(considered),
        "queries_unscorable": sum(1 for q in queries if q["unscorable"]),
        "queries_without_scores": sum(
            1 for q in queries if not q["unscorable"] and q.get("pool_max_rerank_score") is None
        ),
        "hits": _distribution(hit_scores),
        "non_hits": _distribution(non_hit_scores),
        "thresholds": rows,
    }


LABEL_EXPORT_COLUMNS = (
    "query_id", "query", "context", "ranked_query", "rank", "tmdb_id", "title",
    "year", "overview", "rerank_score", "golden_hit", "label",
)


def label_export_rows(queries: list[dict]) -> list[dict]:
    """One row per (query, recommended movie), in rank order, for hand-labelling.

    `queries` are per-query results of a dev run with the gate off. `query` is
    what the user typed and `ranked_query` the text the reranker actually saw
    (the standalone rewrite for a follow-up in oracle mode): the labeller
    judges relevance against `ranked_query`, with `context` for orientation.
    `golden_hit` is informative only -- the golden lists are incomplete, which
    is why a human label column exists. `label` is left empty.
    """
    rows = []
    for q in queries:
        for rank, movie in enumerate(q["recommended"], start=1):
            score = movie.get("rerank_score")
            rows.append({
                "query_id": q["id"],
                "query": q["query"],
                "context": " | ".join(q.get("context", [])),
                "ranked_query": q["query_text"],
                "rank": rank,
                "tmdb_id": movie["tmdb_id"],
                "title": movie["title"],
                "year": (movie.get("release_date") or "")[:4],
                "overview": movie.get("overview") or "",
                "rerank_score": "" if score is None else repr(float(score)),
                "golden_hit": int(bool(movie["hit"])),
                "label": "",
            })
    return rows


def evaluate_query(
    recommended_ids: list[Hashable],
    grades: dict[Hashable, float],
    recommended_embeddings: list[list[float]] | None = None,
    recommended_genres: list[list[str]] | None = None,
    genre_freq: dict[str, float] | None = None,
    judge_scores: list[int | None] | None = None,
    k: int = 5,
) -> dict:
    """Per-query metrics. `grades` maps each resolved golden id to its grade.

    `judge_scores` is aligned with `recommended_ids`; None entries are failed
    judge calls. `llm_relevance` averages the judged ones only, and
    `judge_ndcg` is reported only for a fully judged list -- a partial list
    has no honest nDCG, and the failures are counted by the caller.
    """
    relevant = list(grades)
    metrics = {
        "precision_at_k": precision_at_k(recommended_ids, relevant, k),
        "recall_at_k": recall_at_k(recommended_ids, relevant, k),
        "ndcg_at_k": ndcg_at_k(recommended_ids, grades, k),
    }
    if recommended_embeddings:
        metrics["diversity"] = diversity(recommended_embeddings)
    if recommended_genres and genre_freq:
        metrics["novelty"] = novelty(recommended_genres, genre_freq)
    if judge_scores:
        judged = [s for s in judge_scores[:k] if s is not None]
        if judged:
            metrics["llm_relevance"] = float(np.mean(judged))
        if len(judged) == len(judge_scores[:k]):
            metrics["judge_ndcg"] = judge_ndcg(judged, k)
    return metrics


def aggregate_metrics(per_query_metrics: list[dict]) -> dict:
    if not per_query_metrics:
        return {}
    all_keys = set()
    for m in per_query_metrics:
        all_keys.update(m.keys())
    return {
        key: float(np.mean([m[key] for m in per_query_metrics if key in m]))
        for key in sorted(all_keys)
    }


def compute_genre_frequency(catalog_genres: list[list[str]]) -> dict[str, float]:
    """Compute genre frequency across the catalog for novelty calculation."""
    total = len(catalog_genres)
    if total == 0:
        return {}
    counter = Counter()
    for genres in catalog_genres:
        counter.update(genres)
    return {g: count / total for g, count in counter.items()}
