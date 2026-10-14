"""Sentence-transformers embedding service with lazy-loaded singleton model.

Wraps BAAI/bge-m3 (1024-dim, multilingual). The model loads on first call and
stays in memory for the process lifetime.
"""

import logging

import numpy as np
from django.conf import settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Version of the wording/format of build_movie_embedding_text. Bump it whenever
# the template changes (separators, field order, labels): it is recorded in the
# EmbeddingIndex provenance row, so the bump is what tells check_embedding_index
# that stored vectors were built from different text and need
# `generate_embeddings --all`. The truncation lengths are not part of the
# version -- they live in params.yaml and are recorded separately.
EMBEDDING_TEXT_VERSION = 1

_model = None


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


def get_model():
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info("Embedding model loaded")
    return _model


def encode_texts(texts: list[str]) -> np.ndarray:
    """Batch-encode texts into 1024-dim vectors. Returns numpy array of shape (N, 1024)."""
    model = get_model()
    return model.encode(texts, convert_to_numpy=True)


def encode_query(query: str) -> list[float]:
    """Encode a single query string into a 1024-dim vector (as Python list)."""
    model = get_model()
    embedding = model.encode([query], convert_to_numpy=True)[0]
    return embedding.tolist()


def cosine_similarity(vec_a, vec_b) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))
