"""Resolve runtime profiles without changing the canonical YAML configuration."""
from copy import deepcopy


def _merge(base: dict, overrides: dict) -> dict:
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def resolve_inference_config(params: dict, mode: str) -> dict:
    if mode not in ('local', 'openrouter'):
        raise ValueError(f'INFERENCE_MODE must be local or openrouter (got {mode!r}).')
    overrides = {} if mode == 'local' else params['runtime_profiles']['openrouter']
    return _merge(params, overrides)
