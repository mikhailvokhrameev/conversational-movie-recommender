"""One process-wide assembly point for inference providers.

Concrete adapters are connected here as they are implemented. Construction must
remain free of network requests and local model loading.
"""
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class InferenceBackends:
    llm: object
    embedder: object
    reranker: object


@dataclass(frozen=True)
class _PendingAdapter:
    """Configuration-only placeholder until the concrete adapter is wired."""
    provider: str


def _build_backends() -> InferenceBackends:
    from django.conf import settings

    params = settings.EFFECTIVE_PARAMS
    return InferenceBackends(
        llm=_PendingAdapter(params['llm']['provider']),
        embedder=_PendingAdapter(params['embedding']['provider']),
        reranker=_PendingAdapter(params['reranking']['provider']),
    )


@lru_cache(maxsize=1)
def get_inference_backends() -> InferenceBackends:
    return _build_backends()
