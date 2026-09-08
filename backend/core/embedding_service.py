"""Sentence-transformers embedding service with lazy-loaded singleton model.

Wraps BAAI/bge-m3 (1024-dim, multilingual). The model loads on first call and
stays in memory for the process lifetime.
"""

import logging

import numpy as np
from django.conf import settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_model = None


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
