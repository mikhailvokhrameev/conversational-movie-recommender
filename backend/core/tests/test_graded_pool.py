import csv
import socket
from unittest.mock import Mock

import pytest

from core.graded_pool import RunStore, reuse_grades, shuffled_union, rerank_union


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError('external networking forbidden')
    monkeypatch.setattr(socket, 'create_connection', blocked)
    monkeypatch.setattr(socket.socket, 'connect', blocked)


def test_deduplicate_and_seed():
    assert shuffled_union([[1, 2], [2, 3]], 7) == shuffled_union([[1, 2], [2, 3]], 7)
    assert sorted(shuffled_union([[1, 2], [2, 3]], 7)) == [1, 2, 3]


def test_resume_identity_and_incomplete(tmp_path):
    path = tmp_path / 'pool.csv'
    run = RunStore(path, {'mode': 'local'})
    run.save('q1', {'complete': True})
    run.fail('q2', [{'category': 'Timeout'}])
    resumed = RunStore(path, {'mode': 'local'}, resume=True)
    assert resumed.load('q1')['complete']
    assert resumed.load('q2') is None
    with pytest.raises(ValueError, match='identity'):
        RunStore(path, {'mode': 'cloud'}, resume=True)
    with pytest.raises(ValueError, match='exists'):
        RunStore(path, {'mode': 'local'})


def test_grade_reuse_exact_identity_and_conflicts():
    row = dict(query_id='q', query='exact', context=[], split='dev', tmdb_id=1, rubric_version=1, grade='3')
    assert reuse_grades([dict(row, grade='')], [row])[0]['grade'] == '3'
    for key, value in [('query', 'rewritten'), ('context', ['other']), ('split', 'test'), ('rubric_version', 2)]:
        assert reuse_grades([dict(row, grade='', **{key: value})], [row])[0]['grade'] == ''
    with pytest.raises(ValueError, match='conflicting'):
        reuse_grades([row], [dict(row, grade='0')])
    assert reuse_grades([dict(row, grade='')], [dict(row, grade='', label='1')])[0]['grade'] == ''


def test_batch_alignment():
    adapter = Mock()
    adapter.predict_scores.side_effect = lambda q, docs: [float(d.split('.')[0]) for d in docs]
    candidates = [{'id': i, 'serial_name': str(i)} for i in range(7)]
    scores = rerank_union('q', candidates, adapter, 3)
    assert scores == {str(i): float(i) for i in range(7)}
    assert adapter.predict_scores.call_count == 3


def test_old_binary_preserved(tmp_path):
    from core.graded_pool import read_binary_source, read_grade_source
    source = tmp_path / 'labels.csv'
    original = 'query_id,query,context,tmdb_id,label\nq,exact,prior,1,1\n'
    source.write_text(original)
    evidence = read_binary_source(source)
    assert evidence['kind'] == 'binary_relevance'
    assert evidence['rows'][0]['context'] == 'prior'
    assert 'grade' not in evidence['rows'][0]
    assert read_grade_source(source) == []
    assert source.read_text() == original


def test_history_mismatch_never_reuses():
    row = dict(query_id='q', query='exact', history=[{'role': 'user', 'content': 'x'}], split='dev', tmdb_id=1, rubric_version=1, grade='3')
    assert reuse_grades([dict(row, grade='', history=[])], [row])[0]['grade'] == ''


def test_capture_obscure_and_variants(monkeypatch, settings):
    from types import SimpleNamespace
    from core.graded_pool import capture_query
    settings.EVAL_POOL_EXPORT = {'depth': 10, 'seed': 7, 'rerank_batch_size': 50, 'quality_rrf_weight': 1.0, 'quality_scoring_weight': 0.3}
    candidates = [{'id': 1, 'tmdb_id': 1, 'serial_name': 'one', 'embedding': [1., 0.]}]
    retrieve = Mock(return_value=candidates)
    monkeypatch.setattr('core.candidate_generation.generate_candidates', retrieve)
    monkeypatch.setattr('core.inference_backends.get_inference_backends', lambda: SimpleNamespace(reranker=Mock(predict_scores=lambda q, docs: [0.2]*len(docs))))
    monkeypatch.setattr('core.catalog_bounds.catalog_mean_rating', lambda: 5.)
    monkeypatch.setattr('core.scoring.catalog_mean_rating', lambda: 5.)
    p = SimpleNamespace(intent={}, query_embedding=[1., 0.], query_text='query', semantic_query='semantic')
    result = capture_query({'id': 'q', 'query': 'query', 'split': 'dev', 'community_recs': [1]}, p)
    assert retrieve.call_count == 2
    assert retrieve.call_args_list[0].kwargs['rrf_weights']['quality'] == 0
    assert retrieve.call_args_list[1].kwargs['rrf_weights']['quality'] == 1
    assert result['selected'] == [1]
    assert result['raw_reranker'] == {'1': 0.2}
    retrieve.reset_mock()
    p.intent = {'wants_obscure': True}
    result = capture_query({'id': 'q', 'query': 'query', 'split': 'dev'}, p)
    assert retrieve.call_count == 1
    assert set(result['variants']) == {'configured'}


def test_batch_failure_not_silently_accepted():
    adapter = Mock()
    adapter.predict_scores.return_value = []
    with pytest.raises(ValueError, match='invalid reranker'):
        rerank_union('q', [{'id': 1}], adapter, 50)


@pytest.fixture
def pool_command(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from django.core.management import call_command
    from movies.management.commands.evaluate_scoring import Command
    entries = [{'id': 'q1', 'query': 'exact one', 'split': 'dev', 'context': ['prior']},
               {'id': 'q2', 'query': 'exact two', 'split': 'dev'}]
    source = tmp_path / 'queries.json'
    source.write_text(json.dumps(entries))
    movies = Mock()
    movies.objects.order_by.return_value.values.return_value.iterator.return_value = iter([])
    monkeypatch.setattr('movies.management.commands.evaluate_scoring.Movie', movies)
    monkeypatch.setattr('movies.models.EmbeddingIndex.stored', lambda: {'model': 'pinned'})
    prepare = Mock(side_effect=lambda entry, mode: SimpleNamespace(possible_parse_fallback=False))
    monkeypatch.setattr(Command, '_prepare', prepare)
    def snapshot(entry, prepared):
        return {'complete': True, 'entry': entry, 'selected': [1], 'community_recall': {},
                'variants': {'configured': [{'id': 1, 'tmdb_id': 7, 'serial_name': 'movie'}]}}
    capture = Mock(side_effect=snapshot)
    monkeypatch.setattr('core.graded_pool.capture_query', capture)
    def run(output, **kwargs):
        return call_command('evaluate_scoring', queries=str(source), pool_export=str(output), **kwargs)
    return run, prepare, capture, source


def test_command_failed_query_resume_and_blind_csv(pool_command, tmp_path):
    import json
    from django.core.management.base import CommandError
    run, prepare, capture, source = pool_command
    output = tmp_path / 'pool.csv'
    success = capture.side_effect
    def fail_second(entry, p):
        if entry['id'] == 'q2':
            raise TimeoutError('secret provider details')
        return success(entry, p)
    capture.side_effect = fail_second
    with pytest.raises(CommandError, match='q2 failed'):
        run(output)
    assert not output.exists()
    manifest = json.loads(output.with_suffix('.run').joinpath('manifest.json').read_text())
    assert manifest['status'] == 'failed'
    assert manifest['failures']['q2'] == [{'category': 'TimeoutError'}]
    capture.side_effect = success
    prepare.reset_mock()
    run(output, resume=True)
    assert prepare.call_count == 1
    rows = list(csv.DictReader(output.open()))
    assert len(rows) == 2
    assert set(rows[0]) == {'query_id', 'query', 'split', 'tmdb_id', 'title', 'year', 'overview', 'grade'}
    context = json.loads(output.with_suffix('.run').joinpath('assessment.json').read_text())
    assert context['queries'][0]['context'] == ['prior']
    source.write_text(source.read_text() + '\n')
    prepare.reset_mock()
    with pytest.raises(CommandError, match='identity mismatch'):
        run(output, resume=True)
    prepare.assert_not_called()


def test_command_cross_mode_reuse_does_not_skip_inference(pool_command, tmp_path, monkeypatch):
    run, prepare, capture, source = pool_command
    pilot, full = tmp_path / 'pilot.csv', tmp_path / 'full.csv'
    run(pilot, limit_queries=1)
    rows = list(csv.DictReader(pilot.open()))
    rows[0]['grade'] = '3'
    with pilot.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr('movies.management.commands.evaluate_scoring.inference_metadata', lambda: {'mode': 'openrouter'})
    prepare.reset_mock()
    run(full, reuse_grades=str(pilot))
    assert prepare.call_count == 2
    assert [r['grade'] for r in csv.DictReader(full.open())] == ['3', '']


def test_command_rejects_oracle_before_inference(pool_command, tmp_path):
    from django.core.management.base import CommandError
    run, prepare, capture, source = pool_command
    with pytest.raises(CommandError, match='e2e'):
        run(tmp_path / 'pool.csv', intent='oracle')
    prepare.assert_not_called()


def test_replay_normalizes_selected_head_not_export_batches(monkeypatch, settings):
    from core.graded_pool import replay
    settings.RERANK_TOP_K = 2
    settings.RERANK_WEIGHT = 0.5
    settings.RERANK_MIN_SCORE = None
    settings.EVAL_POOL_EXPORT = {'depth': 10}
    candidates = [{'id': i, 'total': v} for i, v in [(1, 0.9), (2, 0.8), (3, 0.1)]]
    monkeypatch.setattr('core.scoring.score_candidates', lambda *a, **kw: candidates)
    top, head = replay(candidates, [], {}, {'1': -2., '2': -1., '3': 999.}, {})
    assert [c['id'] for c in head] == [1, 2]
    assert [c['total'] for c in top] == [0.5, 0.5]
    assert len(top) == 2  # null gate preserves negative raw reranker scores


@pytest.mark.parametrize('key,value', [('depth', 0), ('rerank_batch_size', False), ('seed', 1.2)])
def test_pool_settings_reject_invalid(key, value):
    from recommender.settings import _validate_pool_export
    with pytest.raises(ValueError, match='pool_export'):
        _validate_pool_export({**{'depth': 10, 'seed': 7, 'rerank_batch_size': 50, 'quality_rrf_weight': 1.0, 'quality_scoring_weight': 0.3}, key: value})


def test_fallback_marks_run_failed(pool_command, tmp_path):
    from types import SimpleNamespace
    from django.core.management.base import CommandError
    run, prepare, capture, source = pool_command
    prepare.side_effect = None
    prepare.return_value = SimpleNamespace(possible_parse_fallback=True)
    with pytest.raises(CommandError, match='failed'):
        run(tmp_path / 'pool.csv')
    capture.assert_not_called()
    assert not (tmp_path / 'pool.csv').exists()
