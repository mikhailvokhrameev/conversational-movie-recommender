"""Provider diagnostics: mocked adapters only, no DB, models or network."""
import json
from io import StringIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from django.conf import settings
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import override_settings

from core.inference_config import resolve_inference_config
from core.llm_client import MessageIntent, aclassify_and_parse
from core.openrouter_client import InferenceError
from movies.tests.test_evaluate_scoring import _Run, _entry


def _backends():
    return SimpleNamespace(
        llm=MagicMock(complete=MagicMock(return_value='OK')),
        embedder=MagicMock(encode_texts=MagicMock(return_value=np.ones((1, 1024)))),
        reranker=MagicMock(predict_scores=MagicMock(return_value=[0.8, 0.2])),
    )


def test_cloud_preflight_uses_actual_adapters_without_ollama():
    backends = _backends()
    cloud = resolve_inference_config(settings.PARAMS, 'openrouter')
    with override_settings(INFERENCE_MODE='openrouter', EFFECTIVE_PARAMS=cloud), \
         patch('core.inference_backends.get_inference_backends', return_value=backends), \
         patch('core.llm_transport.OllamaLLMAdapter', side_effect=AssertionError('no Ollama')):
        out = StringIO()
        call_command('check_inference', stdout=out)
    backends.llm.complete.assert_called_once()
    backends.embedder.encode_texts.assert_called_once()
    backends.reranker.predict_scores.assert_called_once()
    assert 'openrouter' in out.getvalue() and 'voyageai/rerank-3-lite' in out.getvalue()


def test_preflight_invalid_key_is_safe_and_fatal():
    backends = _backends()
    backends.llm.complete.side_effect = InferenceError('post_json', 401)
    with patch('core.inference_backends.get_inference_backends', return_value=backends):
        with pytest.raises(CommandError, match='llm.*401'):
            call_command('check_inference', stdout=StringIO())


def test_effective_metadata_different_modes_and_no_secrets():
    from core.inference_diagnostics import inference_metadata
    records = []
    for mode in ('local', 'openrouter'):
        with override_settings(INFERENCE_MODE=mode,
                               EFFECTIVE_PARAMS=resolve_inference_config(settings.PARAMS, mode),
                               OPENROUTER_API_KEY='secret-key-value'):
            records.append(inference_metadata())
    assert records[0]['effective_config_sha256'] != records[1]['effective_config_sha256']
    assert records[1]['components']['embedding']['model'] == 'BAAI/bge-m3'
    assert records[1]['components']['embedding']['api_model'] == 'baai/bge-m3'
    assert 'secret-key-value' not in json.dumps(records)


@pytest.mark.parametrize('component', ['llm', 'embedding', 'reranking'])
def test_preflight_failures_record_component(component):
    from core.inference_diagnostics import capture_failures, check_inference
    backends = _backends()
    target = {'llm': backends.llm.complete, 'embedding': backends.embedder.encode_texts,
              'reranking': backends.reranker.predict_scores}[component]
    target.side_effect = InferenceError('post_json', 503)
    with patch('core.inference_backends.get_inference_backends', return_value=backends):
        with capture_failures() as failures, pytest.raises(CommandError):
            check_inference()
    assert failures[0]['component'] == component
    assert failures[0]['status_code'] == 503 and failures[0]['latency_ms'] >= 0


def test_evaluation_detects_swallowed_parse_failure(tmp_path):
    run = _Run(tmp_path, [_entry()])
    run.classify.side_effect = aclassify_and_parse
    with patch('core.llm_client._request_classify_and_parse',
               AsyncMock(side_effect=InferenceError('post_json', 401))):
        with pytest.raises(CommandError, match='classify_parse'):
            run()
    record = json.loads(run.output.read_text())
    assert record['status'] == 'failed'
    assert record['inference']['failures'][0]['component'] == 'llm'
    run.pipeline.assert_not_called()


def test_evaluation_detects_swallowed_reranker_failure(tmp_path, caplog):
    from core.reranking import rerank_candidates
    run = _Run(tmp_path, [_entry()])
    movies = run.pipeline.return_value[0]
    def pipeline(*args, **kwargs):
        ranked = rerank_candidates('movie', movies)
        return ranked, ranked
    run.pipeline.side_effect = pipeline
    backends = _backends()
    backends.reranker.predict_scores.side_effect = RuntimeError('secret-upstream-payload')
    with patch('core.reranking.get_inference_backends', return_value=backends):
        with pytest.raises(CommandError, match='rerank'):
            run()
    record = json.loads(run.output.read_text())
    assert record['inference']['failures'][0]['category'] == 'RuntimeError'
    assert 'secret-upstream-payload' not in json.dumps(record) + caplog.text
    assert 'provider=' in caplog.text and 'model=' in caplog.text


def test_evaluation_records_escaping_embedding_failure(tmp_path):
    run = _Run(tmp_path, [_entry()])
    run.encode.side_effect = InferenceError('post_json', 503)
    with pytest.raises(CommandError):
        run()
    record = json.loads(run.output.read_text())
    assert record['status'] == 'failed' and record['inference']['failures']


def test_successful_evaluation_records_effective_providers(tmp_path):
    record = _Run(tmp_path, [_entry()])()
    assert record['status'] == 'ok'
    assert record['inference']['mode'] == settings.INFERENCE_MODE
    assert record['inference']['failures'] == []
    assert len(record['inference']['effective_config_sha256']) == 64


def test_offline_report_needs_no_db_key_or_preflight(tmp_path):
    path = tmp_path / 'labels.csv'
    with override_settings(OPENROUTER_API_KEY=''), \
         patch('movies.management.commands.evaluate_scoring.Command._read_labelled_csv', return_value=[]), \
         patch('movies.management.commands.evaluate_scoring.Command._write_label_report'), \
         patch('movies.management.commands.evaluate_scoring.check_inference', side_effect=AssertionError('preflight')), \
         patch('movies.management.commands.evaluate_scoring._catalog_stats', side_effect=AssertionError('DB')):
        call_command('evaluate_scoring', '--label-report', str(path), stdout=StringIO())


@pytest.mark.asyncio
async def test_embedding_failure_returns_neutral_503_without_scoring():
    from movies.views import ChatView
    intent = MessageIntent(category='new_search', semantic_query='movie', standalone_query='movie')
    session = SimpleNamespace(history=[])
    with patch('movies.views.encode_query', side_effect=InferenceError('post_json', 503)), \
         patch('movies.views._generate_and_score', AsyncMock()) as score:
        response = await ChatView()._handle_search_like(session, 'movie', intent, 1)
    assert response.status_code == 503
    assert json.loads(response.content) == {'error': 'Recommendation service is temporarily unavailable.'}
    score.assert_not_awaited()


def test_usage_logs_only_numeric_usage_and_cost(caplog):
    from core.inference_diagnostics import log_usage
    with caplog.at_level('INFO', logger='core.inference_diagnostics'):
        log_usage('embedding', {'usage': {'prompt_tokens': 3, 'total_tokens': 3, 'cost': 0.02,
                                         'private': 'secret-input'}, 'data': 'secret-output'})
        log_usage('llm', {'usage': {'prompt_tokens': 'secret-key', 'cost': float('nan')}})
    assert 'prompt_tokens=3' in caplog.text and 'cost=0.02' in caplog.text
    assert 'secret-' not in caplog.text and 'nan' not in caplog.text
