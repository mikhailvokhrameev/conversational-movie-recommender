from copy import deepcopy
from unittest.mock import patch

import pytest
from django.conf import settings
from core.inference_config import resolve_inference_config
from core.inference_backends import get_inference_backends


def test_recursive_merge_copy_and_list_replacement():
    params = {'nested': {'keep': 1, 'change': {'a': 2, 'b': [1, 2]}},
              'runtime_profiles': {'openrouter': {'nested': {'change': {'b': [3]}}}}}
    original = deepcopy(params)
    effective = resolve_inference_config(params, 'openrouter')
    assert effective['nested'] == {'keep': 1, 'change': {'a': 2, 'b': [3]}}
    effective['nested']['change']['b'].append(4)
    assert params == original
    local = resolve_inference_config(params, 'local')
    local['nested']['change']['b'].append(5)
    assert params == original


@pytest.mark.parametrize('mode', ['', 'cloud', None])
def test_unknown_mode_rejected(mode):
    with pytest.raises(ValueError, match='INFERENCE_MODE'):
        resolve_inference_config(settings.PARAMS, mode)


def test_shipped_profiles():
    local = resolve_inference_config(settings.PARAMS, 'local')
    cloud = resolve_inference_config(settings.PARAMS, 'openrouter')
    assert local['llm']['model'] == 'qwen3.5:9b-q4_K_M'
    assert cloud['llm']['model'] == 'qwen/qwen3.5-9b'
    assert local['llm']['provider'] == 'ollama'
    assert cloud['llm']['base_url'] == 'https://openrouter.ai/api/v1'
    assert cloud['embedding']['model'] == local['embedding']['model'] == 'BAAI/bge-m3'
    assert cloud['embedding']['api_model'] == 'baai/bge-m3'
    assert cloud['embedding']['dimensions'] == local['embedding']['dimensions'] == 1024
    assert cloud['embedding']['text'] == local['embedding']['text']
    assert cloud['embedding']['revision'] is None
    assert cloud['embedding']['provider_preferences'] == {'order': ['DeepInfra'], 'allow_fallbacks': False}
    assert cloud['reranking']['model'] == 'voyageai/rerank-3-lite'
    assert cloud['reranking']['revision'] is None
    assert cloud['reranking']['min_score'] is None
    assert local['reranking']['min_score'] == 0.005
    assert cloud['reranking']['top_k'] == 50
    assert cloud['reranking']['weight'] == 0.5
    assert cloud['embedding']['timeout_seconds'] == cloud['reranking']['timeout_seconds'] == 60
    assert cloud['llm']['max_output_tokens'] == {'structured': 1024, 'generation': 1024}


def test_settings_default_local_and_neutral_names():
    assert settings.INFERENCE_MODE == 'local'
    assert settings.EFFECTIVE_PARAMS == resolve_inference_config(settings.PARAMS, 'local')
    assert settings.LLM_MODEL == settings.OLLAMA_MODEL
    assert settings.LLM_TIMEOUTS == settings.OLLAMA_TIMEOUTS


def test_assembly_cached():
    get_inference_backends.cache_clear()
    try:
        with patch('core.inference_backends._build_backends') as build:
            result = get_inference_backends()
            assert get_inference_backends() is result
            build.assert_called_once()
    finally:
        get_inference_backends.cache_clear()


def test_assembly_without_key_or_io(monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    get_inference_backends.cache_clear()
    try:
        with patch('socket.socket', side_effect=AssertionError('network forbidden')):
            result = get_inference_backends()
        assert all(getattr(result, role) is not None for role in ('llm', 'embedder', 'reranker'))
    finally:
        get_inference_backends.cache_clear()


def test_cloud_settings_import_without_key(monkeypatch):
    import runpy

    monkeypatch.setenv('INFERENCE_MODE', 'openrouter')
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    import recommender.settings as settings_module
    loaded = runpy.run_path(settings_module.__file__)
    assert loaded['INFERENCE_MODE'] == 'openrouter'
    assert loaded['LLM_MODEL'] == 'qwen/qwen3.5-9b'
    assert loaded['EMBEDDING_MODEL'] == 'BAAI/bge-m3'
    assert loaded['EMBEDDING_REVISION'] is None
    assert loaded['RERANK_MIN_SCORE'] is None
