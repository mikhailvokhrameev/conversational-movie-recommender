"""Cross-encoder reranking of the top scored candidates.

Where the four-signal scorer compares a query *embedding* to a movie
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
XLM-RoBERTa-large backbone: ~302M body parameters. sentence-transformers
loads it in fp32 (it does not downcast this model), and measured timings
on GPU are ~0.26s for 20 pairs, ~0.64s for 50 pairs, and ~1.22s for 100
pairs, versus ~5.6s for 20 pairs on CPU -- the same model is either
comfortably interactive or completely unusable depending on where it runs.

So: on a GPU machine, run this model on the GPU (~2.2GB of VRAM at fp32).
Without a GPU, do not merely move this model to CPU -- switch to a small
reranker such as cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (~21M body
parameters, ~14x less compute, also multilingual), or set
reranking.enabled to false. Both knobs live in params.yaml.

Fail-safe: any error loading or running the model logs and returns the
candidates untouched. A reranker is a quality improvement, never a
correctness dependency, so it must not be able to break a request.
"""

import math
import logging
import threading
import time

from django.conf import settings

from core.inference_diagnostics import inference_operation, log_usage, record_failure
from core.scoring import normalize_scores
from core.inference_backends import get_inference_backends
from core.openrouter_client import InferenceError

logger = logging.getLogger(__name__)

class LocalRerankingAdapter:
    provider = "local"

    def __init__(self):
        self._model = None
        self._load_failed = False
        self._load_lock = threading.Lock()
        self._predict_lock = threading.Lock()

    def get_model(self):
        """Load the cross-encoder once per process. Returns None if unavailable."""

        if self._model is not None or self._load_failed:
            return self._model

        with self._load_lock:
            if self._model is not None or self._load_failed:
                return self._model
            started = time.perf_counter()
            try:
                from sentence_transformers import CrossEncoder

                logger.info(f"Loading reranker model: {settings.RERANKER_MODEL}")
                started = time.perf_counter()
                # "auto" hands device choice to sentence-transformers, which
                # prefers cuda, then mps, then cpu. Explicit values pin it.
                device = settings.RERANKER_DEVICE
                self._model = CrossEncoder(
                    settings.RERANKER_MODEL,
                    revision=settings.RERANKER_REVISION,
                    max_length=settings.RERANK_MAX_LENGTH,
                    device=None if device == "auto" else device,
                )
                logger.info(f"Reranker loaded in {time.perf_counter() - started:.1f}s")
            except Exception as error:
                # Most likely a failed model download. Degrade to no reranking
                # rather than failing every subsequent request.
                self._load_failed = True
                record_failure("reranking", "load_model", error, started)

        return self._model


    def predict_scores(self, query: str, documents: list[str]) -> list[float]:
        model = self.get_model()
        if model is None:
            raise InferenceError("reranker_unavailable")
        # CrossEncoder's native Sigmoid behavior remains local to BGE.
        with self._predict_lock:
            return [float(score) for score in model.predict([(query, doc) for doc in documents])]


class OpenRouterRerankingAdapter:
    provider = "openrouter"

    def predict_scores(self, query: str, documents: list[str]) -> list[float]:
        from core import openrouter_client

        config = settings.EFFECTIVE_PARAMS['reranking']
        body = openrouter_client.post_json('/rerank', {
            'model': config['model'], 'query': query, 'documents': documents,
            'top_n': len(documents),
        }, timeout=config['timeout_seconds'])
        log_usage("reranking", body)
        ordered = None
        try:
            results = body['results']
            if not isinstance(results, list) or len(results) != len(documents):
                raise ValueError
            scores = [None] * len(documents)
            for item in results:
                index, score = item['index'], item['relevance_score']
                if type(index) is not int or not 0 <= index < len(scores) or scores[index] is not None:
                    raise ValueError
                if type(score) not in (int, float) or not math.isfinite(score):
                    raise ValueError
                scores[index] = float(score)
            ordered = scores
        except (KeyError, TypeError, ValueError, IndexError, OverflowError):
            pass
        if ordered is None:
            raise InferenceError("rerank_response")
        return ordered


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

    try:
        started = time.perf_counter()
        with inference_operation("reranking", "rerank"):
            raw_scores = get_inference_backends().reranker.predict_scores(
                query, [_movie_text(candidate) for candidate in head],
            )
            if len(raw_scores) != len(head) or any(not math.isfinite(score) for score in raw_scores):
                raise InferenceError("rerank_response")
        logger.info("Reranked %d candidates in %.2fs", len(head), time.perf_counter() - started)
    except Exception:
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
