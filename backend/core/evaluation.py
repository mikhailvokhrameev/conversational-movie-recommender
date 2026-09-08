"""Offline evaluation metrics for recommendation quality.

Three categories of metrics:
  Accuracy:        precision@k, recall@k, ndcg@k (with optional graded relevance)
  Beyond-accuracy: diversity, novelty, coverage
  LLM-as-judge:    graded relevance scoring via Ollama (opt-in, slow)

Usage:
  from core.evaluation import evaluate_query, aggregate_metrics
  metrics = evaluate_query(recommended, relevant, embeddings, k=5)
"""

import json
import logging
import math
from collections import Counter

import httpx
import numpy as np

from core.embedding_service import cosine_similarity
from core.ollama_client import _chat_sync

logger = logging.getLogger(__name__)

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


def precision_at_k(recommended: list[str], relevant: list[str], k: int = 5) -> float:
    if k <= 0:
        return 0.0
    top_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / k


def recall_at_k(recommended: list[str], relevant: list[str], k: int = 5) -> float:
    if not relevant:
        return 0.0
    top_k = set(recommended[:k])
    relevant_set = set(relevant)
    hits = len(top_k & relevant_set)
    return hits / len(relevant_set)


def ndcg_at_k(
    recommended: list[str],
    relevant: list[str],
    k: int = 5,
    graded_scores: list[float] | None = None,
) -> float:
    """NDCG with optional graded relevance (from LLM-as-judge scores)."""
    if graded_scores:
        dcg = sum(
            score / math.log2(i + 2) for i, score in enumerate(graded_scores[:k])
        )
        ideal_scores = sorted(graded_scores, reverse=True)[:k]
        idcg = sum(
            score / math.log2(i + 2) for i, score in enumerate(ideal_scores)
        )
        return dcg / idcg if idcg > 0 else 0.0

    relevant_set = set(relevant)
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, item in enumerate(recommended[:k])
        if item in relevant_set
    )
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def coverage(all_recommendations: list[list[str]], catalog_size: int) -> float:
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


def llm_relevance_score(query: str, movie: dict) -> int:
    """Ask Ollama to rate relevance of a movie to a query on a 1-5 scale."""
    prompt = LLM_JUDGE_PROMPT.format(
        query=query,
        title=movie.get("serial_name", ""),
        genres=", ".join(movie.get("genres", [])),
        description=(movie.get("description", "") or "")[:300],
    )
    try:
        content = _chat_sync(
            [{"role": "user", "content": prompt}],
            json_mode=True,
            timeout=30.0,
        )
        score = int(json.loads(content).get("score", 3))
        return max(1, min(5, score))
    except (httpx.HTTPError, json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning(f"LLM judge failed for '{movie.get('serial_name', '')}': {e}")
        return 3


def evaluate_query(
    recommended_titles: list[str],
    relevant_titles: list[str],
    recommended_embeddings: list[list[float]] | None = None,
    recommended_genres: list[list[str]] | None = None,
    genre_freq: dict[str, float] | None = None,
    graded_scores: list[float] | None = None,
    k: int = 5,
) -> dict:
    metrics = {
        "precision_at_k": precision_at_k(recommended_titles, relevant_titles, k),
        "recall_at_k": recall_at_k(recommended_titles, relevant_titles, k),
        "ndcg_at_k": ndcg_at_k(recommended_titles, relevant_titles, k, graded_scores),
    }
    if recommended_embeddings:
        metrics["diversity"] = diversity(recommended_embeddings)
    if recommended_genres and genre_freq:
        metrics["novelty"] = novelty(recommended_genres, genre_freq)
    if graded_scores:
        metrics["llm_relevance"] = float(np.mean(graded_scores))
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
