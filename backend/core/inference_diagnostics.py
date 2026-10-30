"""Safe provider-neutral diagnostics, scoped to one request or evaluation run."""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import hashlib
import json
import logging
import math
import time

import numpy as np
from django.conf import settings
from django.core.management.base import CommandError

from core.openrouter_client import InferenceError

logger = logging.getLogger(__name__)
_failures = ContextVar('inference_failures', default=None)
_seen = ContextVar('inference_seen_errors', default=None)


def inference_metadata():
    params = settings.EFFECTIVE_PARAMS
    return {
        'mode': settings.INFERENCE_MODE,
        'effective_config_sha256': hashlib.sha256(
            json.dumps(params, sort_keys=True, separators=(',', ':')).encode(),
        ).hexdigest(),
        'components': {
            name: {key: params[name][key] for key in ('provider', 'model', 'api_model', 'revision')
                   if key in params[name]}
            for name in ('llm', 'embedding', 'reranking')
        },
    }


def log_usage(component, body):
    """Allowlist numeric provider usage; never include arbitrary response fields."""
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return
    fields = []
    for name in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens", "cost"):
        value = usage.get(name)
        if (type(value) in (int, float) and value >= 0 and math.isfinite(value)
                and (name == "cost" or type(value) is int)):
            fields.append(f"{name}={value}")
    if fields:
        config = settings.EFFECTIVE_PARAMS[component]
        logger.info("Inference usage component=%s provider=%s model=%s %s", component,
                    config["provider"], config.get("api_model", config["model"]), " ".join(fields))


@contextmanager
def capture_failures():
    failures = []
    token = _failures.set(failures)
    seen_token = _seen.set(set())
    try:
        yield failures
    finally:
        _seen.reset(seen_token)
        _failures.reset(token)


def record_failure(component, operation, error, started=None):
    """Keep categories/status only; exception text may contain credentials or input."""
    seen = _seen.get()
    if seen is not None:
        if id(error) in seen:
            return
        seen.add(id(error))
    config = settings.EFFECTIVE_PARAMS[component]
    failure = {
        'component': component, 'operation': operation,
        'provider': config['provider'], 'model': config.get('api_model', config['model']),
        'category': type(error).__name__,
        'status_code': error.status_code if isinstance(error, InferenceError) else None,
        'latency_ms': max(0, (time.perf_counter() - started) * 1000) if started is not None else 0,
    }
    failures = _failures.get()
    if failures is not None:
        failures.append(failure)
    logger.warning(
        'Inference failed component=%s operation=%s provider=%s model=%s category=%s status=%s latency_ms=%.1f',
        *(failure[key] for key in ('component', 'operation', 'provider', 'model', 'category',
                                  'status_code', 'latency_ms')),
    )


@contextmanager
def inference_operation(component, operation):
    started = time.perf_counter()
    try:
        yield
    except Exception as error:
        record_failure(component, operation, error, started)
        raise
    else:
        config = settings.EFFECTIVE_PARAMS[component]
        logger.info('Inference operation=%s provider=%s model=%s latency_ms=%.1f',
                    operation, config['provider'], config.get('api_model', config['model']),
                    (time.perf_counter() - started) * 1000)


def require_healthy():
    failures = _failures.get()
    if failures:
        failure = failures[-1]
        raise CommandError(
            f"Inference failed: {failure['component']} {failure['operation']} "
            f"({failure['category']}, status={failure['status_code']})."
        )


def check_inference():
    """Probe real adapters with small inputs. Only users run this against providers."""
    from core.inference_backends import get_inference_backends

    # Reuse the evaluation scope, or establish a standalone command scope.
    scope = capture_failures() if _failures.get() is None else nullcontext(_failures.get())
    with scope:
        try:
            backends = get_inference_backends()
            with inference_operation('llm', 'preflight'):
                content = backends.llm.complete(
                    [{'role': 'user', 'content': 'Reply OK.'}], schema=None,
                    kind='generation', timeout=settings.LLM_TIMEOUTS['intent'], call_type='preflight',
                )
                if not isinstance(content, str) or not content.strip():
                    raise InferenceError('llm_response')
            with inference_operation('embedding', 'preflight'):
                vectors = np.asarray(backends.embedder.encode_texts(['A movie.']))
                if (vectors.shape != (1, settings.EMBEDDING_DIMENSIONS)
                        or not np.isfinite(vectors).all() or not np.any(vectors)):
                    raise InferenceError('embedding_response')
            if settings.EFFECTIVE_PARAMS['reranking']['enabled']:
                with inference_operation('reranking', 'preflight'):
                    scores = backends.reranker.predict_scores('movie', ['A movie.', 'A book.'])
                    if len(scores) != 2 or not np.isfinite(scores).all():
                        raise InferenceError('rerank_response')
        except Exception:
            require_healthy()
            raise
    return inference_metadata()
