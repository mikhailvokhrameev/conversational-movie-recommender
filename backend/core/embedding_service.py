"""Shared embedding interface and provider adapters; local weights load lazily."""

import logging

import numpy as np
from django.conf import settings

logger = logging.getLogger(__name__)

# Version of the wording/format of build_movie_embedding_text. Bump it whenever
# the template changes (separators, field order, labels): it is recorded in the
# EmbeddingIndex provenance row, so the bump is what tells check_embedding_index
# that stored vectors were built from different text and need
# `generate_embeddings --all`. The truncation lengths are not part of the
# version -- they live in params.yaml and are recorded separately.
EMBEDDING_TEXT_VERSION = 1



def build_movie_embedding_text(
    name,
    genres,
    description,
    keywords,
    description_chars=None,
    max_keywords=None,
) -> str:
    """The text embedded for one movie; generate_embeddings is its only caller.

    Truncation lengths default to params.yaml. None for genres, description or
    keywords (a null catalog column) is treated as empty.
    """
    if description_chars is None:
        description_chars = settings.EMBEDDING_TEXT_DESCRIPTION_CHARS
    if max_keywords is None:
        max_keywords = settings.EMBEDDING_TEXT_MAX_KEYWORDS
    genre_str = ", ".join(genres) if genres else ""
    desc_truncated = (description or "")[:description_chars]
    keyword_str = ", ".join((keywords or [])[:max_keywords])
    return f"{name}. {genre_str}. {desc_truncated}. Keywords: {keyword_str}"


class LocalEmbeddingAdapter:
    """Preserve the local model's native output and normalization semantics."""

    provider = "local"

    def __init__(self):
        self._model = None

    def get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL)
            self._model = SentenceTransformer(
                settings.EMBEDDING_MODEL, revision=settings.EMBEDDING_REVISION,
            )
            logger.info("Embedding model loaded")
        return self._model

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, settings.EMBEDDING_DIMENSIONS))
        return self.get_model().encode(texts, convert_to_numpy=True)


def _ordered_vectors(body, count):
    """Validate the entire remote batch before exposing any vectors."""
    from core.openrouter_client import InferenceError

    result = None
    try:
        data = body['data']
        if not isinstance(data, list) or len(data) != count:
            raise ValueError
        ordered = [None] * count
        for item in data:
            index = item['index']
            if type(index) is not int or not 0 <= index < count or ordered[index] is not None:
                raise ValueError
            vector = item['embedding']
            if not isinstance(vector, list) or len(vector) != settings.EMBEDDING_DIMENSIONS:
                raise ValueError
            if any(type(value) not in (int, float) for value in vector):
                raise ValueError
            vector = np.asarray(vector, dtype=float)
            if not np.isfinite(vector).all() or not np.any(vector):
                raise ValueError
            ordered[index] = vector
        result = np.asarray(ordered)
    except (KeyError, TypeError, ValueError, IndexError, OverflowError):
        pass
    # Do not retain upstream payloads in exception messages or contexts.
    if result is None:
        raise InferenceError("embedding_response")
    return result


class OpenRouterEmbeddingAdapter:
    provider = "openrouter"

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        from core import openrouter_client

        if not texts:
            return np.empty((0, settings.EMBEDDING_DIMENSIONS))
        config = settings.EFFECTIVE_PARAMS['embedding']
        batches = []
        for offset in range(0, len(texts), settings.EMBEDDING_BATCH_SIZE):
            batch = texts[offset:offset + settings.EMBEDDING_BATCH_SIZE]
            payload = {"model": config['api_model'], "input": batch,
                       "provider": config['provider_preferences']}
            body = openrouter_client.post_json(
                '/embeddings', payload, timeout=config['timeout_seconds'],
            )
            batches.append(_ordered_vectors(body, len(batch)))
        return np.concatenate(batches)


def encode_texts(texts: list[str]) -> np.ndarray:
    """Encode through the process-wide provider selection, in input order."""
    from core.inference_backends import get_inference_backends

    if not texts:
        return np.empty((0, settings.EMBEDDING_DIMENSIONS))
    return get_inference_backends().embedder.encode_texts(texts)


def encode_query(query: str) -> list[float]:
    """Encode the original query without adding provider-specific prefixes."""
    return encode_texts([query])[0].tolist()


def cosine_similarity(vec_a, vec_b) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))
