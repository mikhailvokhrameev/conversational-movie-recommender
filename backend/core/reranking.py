"""Cross-encoder reranking of the top scored candidates.

Where the three-signal scorer compares a query *embedding* to a movie
*embedding* -- two vectors produced independently, never seeing each other --
a cross-encoder reads the query and the movie text together in one forward
pass and scores their relevance directly. That joint view is what lets it
catch relationships bi-encoders miss, and it is why reranking a short list
usually beats any amount of weight tuning on the list's ordering.

The cost is that it cannot be precomputed: every (query, movie) pair is a
model call, so it only runs on a short slice of already-good candidates.

Pipeline position:
  candidates ──> score_candidates ──> rerank (top-K) ──> mmr_diversify ──> top-N

Model choice and device are coupled, and getting the pair wrong is the main
way this step goes bad. bge-reranker-v2-m3 (the configured default) is an
XLM-RoBERTa-large backbone: ~302M body parameters, ~155 GFLOPs per pair at
256 tokens, so a 20-candidate rerank is ~3.1 TFLOPs. That is roughly 0.15s
on a discrete GPU and roughly 15s on CPU -- the same model is either
comfortably interactive or completely unusable depending on where it runs.

So: on a GPU machine, run this model on the GPU (~1.1GB of VRAM at fp16).
Without a GPU, do not merely move this model to CPU -- switch to a small
reranker such as cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (~21M body
parameters, ~14x less compute, also multilingual), or set
reranking.enabled to false. Both knobs live in params.yaml.

Fail-safe: any error loading or running the model logs and returns the
candidates untouched. A reranker is a quality improvement, never a
correctness dependency, so it must not be able to break a request.
"""

import logging
import threading
import time

from django.conf import settings

from core.scoring import normalize_scores

logger = logging.getLogger(__name__)

_model = None
_load_failed = False
_load_lock = threading.Lock()

# Serializes inference. A predict() call already uses the whole device -- all
# cores on CPU, or a full batch on GPU -- so concurrent calls contend rather
# than overlap, and on GPU they also stack VRAM for no throughput gain.
_predict_lock = threading.Lock()



def get_model():
    """Load the cross-encoder once per process. Returns None if unavailable."""
    global _model, _load_failed

    if _model is not None or _load_failed:
        return _model

    with _load_lock:
        if _model is not None or _load_failed:
            return _model
        try:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading reranker model: {settings.RERANKER_MODEL}")
            started = time.perf_counter()
            # "auto" hands device choice to sentence-transformers, which
            # prefers cuda, then mps, then cpu. Explicit values pin it.
            device = settings.RERANKER_DEVICE
            _model = CrossEncoder(
                settings.RERANKER_MODEL,
                max_length=settings.RERANK_MAX_LENGTH,
                device=None if device == "auto" else device,
            )
            logger.info(f"Reranker loaded in {time.perf_counter() - started:.1f}s")
        except Exception:
            # Most likely a failed model download. Degrade to no reranking
            # rather than failing every subsequent request.
            _load_failed = True
            logger.exception("Reranker unavailable, continuing without it")

    return _model


def rerank_candidates(
    query: str,
    candidates: list[dict],
    top_k: int | None = None,
    weight: float | None = None,
) -> list[dict]:
    """Rerank the best `top_k` candidates and return just those, reordered.

    Returns a *truncated* list: candidates below the top_k cut are dropped,
    since they ranked below `top_k` items and cannot reach a top-5 result.
    Keeping them would also mean mixing reranked and un-reranked `total`
    values in one list, which are no longer on the same scale.

    The cross-encoder score is blended with the existing scorer total rather
    than replacing it -- the scorer carries session personalization and genre
    intent, which the cross-encoder knows nothing about. Both are min-max
    normalized across the slice before blending, for the same reason the
    scorer normalizes its own signals: without it the blend weight would not
    mean what it says.

    Returns the full candidate pool if reranking is disabled or there is
    nothing to do (fewer than 2 candidates); if the model is unavailable or
    prediction fails, returns the full pool re-sorted by scorer `total` --
    untruncated in both cases, so a reranker failure costs relevance, not
    candidates the caller would otherwise have had for diversification.
    """
    if not settings.RERANKER_ENABLED or not query or not candidates:
        return candidates

    top_k = top_k if top_k is not None else settings.RERANK_TOP_K
    weight = weight if weight is not None else settings.RERANK_WEIGHT

    ordered = sorted(candidates, key=lambda c: c["total"], reverse=True)
    head = ordered[:top_k]
    if len(head) < 2:
        return head

    model = get_model()
    if model is None:
        return ordered

    pairs = [(query, _movie_text(candidate)) for candidate in head]
    try:
        started = time.perf_counter()
        with _predict_lock:
            raw_scores = model.predict(pairs)
        elapsed = time.perf_counter() - started
        logger.info(f"Reranked {len(pairs)} candidates in {elapsed:.2f}s")
    except Exception:
        logger.exception("Reranking failed, falling back to scorer order")
        return ordered

    rerank_norm = normalize_scores([float(score) for score in raw_scores])
    total_norm = normalize_scores([candidate["total"] for candidate in head])

    reranked = [
        {
            **candidate,
            "total": weight * rerank_norm[i] + (1 - weight) * total_norm[i],
            "rerank_score": float(raw_scores[i]),
            "pre_rerank_total": candidate["total"],
        }
        for i, candidate in enumerate(head)
    ]
    reranked.sort(key=lambda c: c["total"], reverse=True)
    return reranked


def _movie_text(movie: dict) -> str:
    """Render a movie as the text side of the (query, document) pair."""
    parts = [
        movie.get("serial_name") or "",
        ", ".join(movie.get("genres") or []),
        (movie.get("description") or "")[:settings.RERANK_DESCRIPTION_CHARS],
    ]
    return ". ".join(part for part in parts if part)
