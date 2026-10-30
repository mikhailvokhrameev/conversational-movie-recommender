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


def _build_backends() -> InferenceBackends:
    from django.conf import settings

    from core.llm_transport import OllamaLLMAdapter, OpenRouterLLMAdapter

    from core.embedding_service import LocalEmbeddingAdapter, OpenRouterEmbeddingAdapter

    from core.reranking import LocalRerankingAdapter, OpenRouterRerankingAdapter

    params = settings.EFFECTIVE_PARAMS
    llm_adapters = {"ollama": OllamaLLMAdapter, "openrouter": OpenRouterLLMAdapter}
    embedding_adapters = {"local": LocalEmbeddingAdapter, "openrouter": OpenRouterEmbeddingAdapter}
    reranking_adapters = {"local": LocalRerankingAdapter, "openrouter": OpenRouterRerankingAdapter}
    return InferenceBackends(
        llm=llm_adapters[params['llm']['provider']](),
        embedder=embedding_adapters[params['embedding']['provider']](),
        reranker=reranking_adapters[params['reranking']['provider']](),
    )


@lru_cache(maxsize=1)
def get_inference_backends() -> InferenceBackends:
    return _build_backends()
