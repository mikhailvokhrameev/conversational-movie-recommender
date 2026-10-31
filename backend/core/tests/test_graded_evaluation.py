"""Offline replay: socket connections and database cursors are forbidden."""
import copy
import socket
from unittest.mock import Mock

import pytest


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*a, **kw):
        raise AssertionError('network/DB forbidden')
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setattr(socket, 'create_connection', blocked)
    monkeypatch.setattr('django.db.backends.base.base.BaseDatabaseWrapper.cursor', blocked)


def config():
    from django.conf import settings
    return copy.deepcopy(settings.EFFECTIVE_PARAMS)


def snapshot(q='q', split='dev'):
    candidates = [dict(id=i, tmdb_id=i, serial_name=str(i), embedding=[1., float(i)],
                       genres=[], popularity=float(i), vote_average=float(10-i), vote_count=500)
                  for i in range(1, 8)]
    return dict(complete=True, entry=dict(id=q, query=q, split=split, community_recs=[1]),
                intent={}, query_embedding=[1., 0.], catalog_mean=5.,
                variants={'configured': candidates, 'quality_1': candidates},
                raw_reranker={str(i): float(i) for i in range(1, 8)}, selected=list(range(1, 8)))


@pytest.mark.parametrize('mode', ['local', 'openrouter'])
def test_replay_matches_production_and_changed_head(monkeypatch, settings, mode):
    from core.graded_evaluation import replay_ranking
    from core.pipeline import generate_and_score
    s, p = snapshot(), config()
    if mode == 'openrouter':
        p['reranking'].update(p['runtime_profiles']['openrouter']['reranking'])
        settings.RERANK_MIN_SCORE = p['reranking']['min_score']
        settings.RERANK_TOP_K = p['reranking']['top_k']
        settings.RERANK_WEIGHT = p['reranking']['weight']
    monkeypatch.setattr('core.scoring.catalog_mean_rating', lambda: 5.)
    monkeypatch.setattr('core.pipeline.generate_candidates', lambda *a, **kw: s['variants']['configured'])
    # Use the production scorer/reranker with a mocked adapter, for both modes.
    adapter = Mock()
    adapter.predict_scores.side_effect = lambda query, docs: [float(d.split('.')[0]) for d in docs]
    monkeypatch.setattr('core.reranking.get_inference_backends', lambda: Mock(reranker=adapter))
    expected, _ = generate_and_score(s['query_embedding'], s['intent'], query_text='q', top_n=5)
    assert [c['id'] for c in replay_ranking(s, p, 5)] == [c['id'] for c in expected]
    p['reranking']['top_k'] = 2
    assert len(replay_ranking(s, p, 5)) <= 2


def test_reference_obscure_null_gate_and_full_pool():
    from core.graded_evaluation import replay_ranking
    s, p = snapshot(), config()
    s['intent'] = {'reference_films': ['7'], 'wants_obscure': True}
    p['reranking'].update(min_score=None, top_k=3)
    a = replay_ranking(s, p, 5)
    p['scoring']['weights'] = dict(semantic=.05, metadata=.05, popularity=0., quality=.9)
    b = replay_ranking(s, p, 5)
    assert all(c['id'] != 7 for c in a)
    assert len(b) <= 3
    # Saved WR is recomputed, and normalization sees all seven, not just head.
    assert max(c['raw_scores']['quality'] for c in b) <= 7


def test_metrics_unjudged_and_small_queries():
    from core.graded_evaluation import grade_report
    s, p = snapshot(), config()
    p['reranking']['min_score'] = None
    grades = {('q', str(i)): 3 for i in range(1, 8)}
    r = grade_report([s], grades, p, 5)
    assert r['ordinary']['ndcg@5'] == pytest.approx(1)
    assert r['ordinary']['good@5'] == 1
    assert r['complete_coverage']
    r = grade_report([s], {('q', '1'): 0, ('q', '2'): 3}, p, 5)
    assert not r['complete_coverage']
    assert r['missing_pairs']
    assert r['ordinary']['ndcg@5'] is None
    assert grade_report([s], {('q', '1'): 3}, p, 5)['skipped_queries'] == 1


def test_fit_grouped_deterministic_and_test_rejection():
    from core.graded_evaluation import fit_ranking_params, grouped_folds
    p = config()
    ss = [snapshot(str(i)) for i in range(5)]
    grades = {(str(q), str(i)): 3 for q in range(5) for i in range(1, 8)}
    assert grouped_folds(ss) == grouped_folds(ss)
    assert len(set(q for fold in grouped_folds(ss) for q in fold)) == 5
    r = fit_ranking_params(ss, grades, p)
    assert r['configurations_evaluated'] == 8
    assert r['applied']
    assert len(r['cross_validation']['folds']) == 5
    with pytest.raises(ValueError, match='five'):
        fit_ranking_params(ss[:4], grades, p)
    with pytest.raises(ValueError, match='test'):
        fit_ranking_params([snapshot(split='test')], grades, p)


def test_csv_errors_and_blanks(tmp_path):
    from core.graded_evaluation import read_grades
    path = tmp_path / 'grades.csv'
    path.write_text('query_id,tmdb_id,grade\nq,1,\nq,2,3\n')
    assert read_grades(path, [snapshot()]) == {('q', '2'): 3}
    for rows, error in [('q,1,4\n', 'line 2'), ('q,99,0\n', 'line 2'), ('q,1,0\nq,1,1\n', 'line 3')]:
        path.write_text('query_id,tmdb_id,grade\n' + rows)
        with pytest.raises(ValueError, match=error):
            read_grades(path, [snapshot()])


def test_full_pool_normalization_and_head_changes():
    from core.graded_evaluation import replay_ranking
    s, p = snapshot(), config()
    p['reranking'].update(top_k=2, weight=0., min_score=None)
    p['scoring']['weights'] = dict(semantic=0., metadata=0., popularity=1., quality=0.)
    top = replay_ranking(s, p, 5)
    assert [c['id'] for c in top] == [7, 6]
    assert top[1]['popularity'] == pytest.approx(5/6)
    p['scoring']['weights'] = dict(semantic=0., metadata=0., popularity=0., quality=1.)
    assert [c['id'] for c in replay_ranking(s, p, 5)] == [1, 2]
    s['intent']['wants_obscure'] = True
    p['scoring']['weights'] = dict(semantic=0., metadata=0., popularity=.1, quality=.9)
    assert [c['id'] for c in replay_ranking(s, p, 5)] == [7, 6]


def test_hand_calculated_metrics_and_no_false_complete():
    from core.graded_evaluation import grade_report
    s, p = snapshot(), config()
    p['reranking'].update(top_k=5, weight=1., min_score=None)
    # semantic-only head is 1..5, reranker reverses to 5..1, no MMR at <= k.
    p['scoring']['weights'] = dict(semantic=1., metadata=0., popularity=0., quality=0.)
    g = {('q', str(i)): v for i, v in enumerate([3, 2, 1, 0, 0, 0, 0], 1)}
    r = grade_report([s], g, p, 5)['ordinary']
    assert r['ndcg@5'] == pytest.approx((1/2+3/2.32192809489+7/2.58496250072)/(7+3/1.58496250072+1/2))
    assert r['junk@5'] == .4
    assert r['good@5'] == .4
    assert r['community_recall'] == 1.


def test_fit_missing_grades_and_gate_evidence():
    from core.graded_evaluation import fit_ranking_params, calibrate_gate
    ss, p = [snapshot(str(i)) for i in range(5)], config()
    grades = {(str(q), str(i)): 3 for q in range(5) for i in [1, 2]}
    assert not fit_ranking_params(ss, grades, p)['applied']
    assert not calibrate_gate(ss, grades, p)['calibrated']
    grades = {(str(q), str(i)): 3 for q in range(5) for i in range(1, 8)}
    result = calibrate_gate(ss, grades, p)
    assert result['calibrated'] and result['threshold'] == 1.


def write_run(tmp_path, name, split, grade='3', **overrides):
    import json
    from core.graded_pool import RunStore, COLUMNS
    ss = [snapshot(str(i), split) for i in range(5 if split == 'dev' else 1)]
    path = tmp_path / (name + '.csv')
    identity = {'inference': {'mode': 'local', 'components': {'reranking': {'model': 'bge'}}},
                'baseline_params': config(), 'catalog_sha256': 'catalog', 'index': {'model': 'embed'},
                'queries': [s['entry'] for s in ss]}
    identity.update(overrides)
    run = RunStore(path, identity)
    for item in ss:
        run.save(item['entry']['id'], item)
    run.manifest['status'] = 'complete'
    run.manifest_path.write_text(json.dumps(run.manifest))
    path.write_text(','.join(COLUMNS)+'\n'+''.join(
        f"{item['entry']['id']},{item['entry']['query']},{split},{i},{i},,,{grade}\n"
        for item in ss for i in range(1, 8)))
    return path


def test_independent_dev_test_proposal_and_freeze(tmp_path):
    from core.graded_evaluation import offline_report
    dev = write_run(tmp_path, 'dev', 'dev')
    test = write_run(tmp_path, 'test', 'test')
    result = offline_report(dev, 'dev', True)
    assert result['fit']['applied']
    assert 'baseline gate' in result['fit']['cross_validation']['scope']
    proposal = dev.with_suffix('.run')/'frozen-fit.json'
    report = offline_report(test, 'test', frozen_proposal=proposal)
    assert report['frozen_choice'] and report['comparison_complete']
    assert report['source_dev_identity_sha256']
    with pytest.raises(ValueError, match='test access'):
        offline_report(dev, 'dev', True)


def test_test_requires_choice_before_reading_grades(tmp_path, monkeypatch):
    from core.graded_evaluation import offline_report
    path = write_run(tmp_path, 'test', 'test')
    monkeypatch.setattr('core.graded_evaluation.read_grades', Mock(side_effect=AssertionError('grades inspected')))
    with pytest.raises(ValueError, match='frozen proposal'):
        offline_report(path, 'test')


@pytest.mark.parametrize('change', ['mode', 'provider', 'catalog', 'index', 'baseline'])
def test_proposal_compatibility_before_grades(tmp_path, monkeypatch, change):
    from core.graded_evaluation import offline_report
    dev = write_run(tmp_path, 'dev', 'dev')
    offline_report(dev, 'dev', True)
    overrides = {}
    if change in ('mode', 'provider'):
        overrides['inference'] = {'mode': 'openrouter' if change == 'mode' else 'local',
                                 'components': {'reranking': {'model': 'voyage'}}}
    elif change == 'catalog':
        overrides['catalog_sha256'] = 'different'
    elif change == 'index':
        overrides['index'] = {'model': 'other'}
    else:
        overrides['baseline_params'] = config()
        overrides['baseline_params']['reranking']['weight'] = .9
    test = write_run(tmp_path, 'test', 'test', **overrides)
    monkeypatch.setattr('core.graded_evaluation.read_grades', Mock(side_effect=AssertionError('grades inspected')))
    with pytest.raises(ValueError, match='incompatible'):
        offline_report(test, 'test', frozen_proposal=dev.with_suffix('.run')/'frozen-fit.json')


@pytest.mark.parametrize('grade', ['', '4'])
def test_first_test_access_locks_even_incomplete_or_invalid_grades(tmp_path, monkeypatch, grade):
    from core.graded_evaluation import offline_report
    dev = write_run(tmp_path, 'dev', 'dev')
    offline_report(dev, 'dev', True)
    proposal = dev.with_suffix('.run')/'frozen-fit.json'
    test = write_run(tmp_path, 'test', 'test', grade=grade)
    original = __import__('core.graded_evaluation', fromlist=['read_grades']).read_grades
    def checked(*args, **kw):
        assert (proposal.parent/'test-access.json').exists()
        return original(*args, **kw)
    monkeypatch.setattr('core.graded_evaluation.read_grades', checked)
    if grade == '4':
        with pytest.raises(ValueError, match='line 2'):
            offline_report(test, 'test', frozen_proposal=proposal)
    else:
        report = offline_report(test, 'test', frozen_proposal=proposal)
        assert not report['comparison_complete']
    with pytest.raises(ValueError, match='test access'):
        offline_report(dev, 'dev', True)


def test_dev_fit_rejects_populated_combined_test_grades(tmp_path):
    import json
    from core.graded_evaluation import offline_report
    from core.graded_pool import digest, atomic_json
    dev = write_run(tmp_path, 'dev', 'dev')
    directory = dev.with_suffix('.run')
    manifest = json.loads((directory/'manifest.json').read_text())
    test = snapshot('t', 'test')
    manifest['identity']['queries'].append(test['entry'])
    atomic_json(directory/'manifest.json', manifest)
    atomic_json(directory/(digest('t')+'.json'), test)
    with dev.open('a') as stream:
        stream.write('t,t,test,1,1,,,3\n')
    with pytest.raises(ValueError, match='held-out test grades'):
        offline_report(dev, 'dev', True)


def test_mmr_and_stable_ties():
    from core.graded_evaluation import replay_ranking
    s, p = snapshot(), config()
    p['reranking'].update(enabled=False, min_score=None)
    p['scoring']['weights'] = dict(semantic=0., metadata=1., popularity=0., quality=0.)
    p['diversification']['lambda'] = .1
    s['variants']['configured'][0]['embedding'] = [1., 0.]
    s['variants']['configured'][1]['embedding'] = [1., 0.]
    s['variants']['configured'][2]['embedding'] = [0., 1.]
    top = replay_ranking(s, p, 2)
    assert [c['id'] for c in top] == [1, 3]
    p['diversification']['lambda'] = 1.
    assert [c['id'] for c in replay_ranking(s, p, 2)] == [1, 2]


@pytest.mark.parametrize('change', ['identity', 'choice'])
def test_copied_proposal_preserves_source_dev_binding(tmp_path, monkeypatch, change):
    import json
    from core.graded_evaluation import offline_report
    from core.graded_pool import atomic_json
    dev = write_run(tmp_path, 'dev', 'dev')
    offline_report(dev, 'dev', True)
    directory = dev.with_suffix('.run')
    proposal = directory/'frozen-fit.json'
    copied = tmp_path/'copied-proposal.json'
    copied.write_bytes(proposal.read_bytes())
    if change == 'identity':
        manifest = json.loads((directory/'manifest.json').read_text())
        manifest['identity']['catalog_sha256'] = 'changed'
        atomic_json(directory/'manifest.json', manifest)
    else:
        choice = json.loads(proposal.read_text())
        choice['params']['reranking']['weight'] = .99
        atomic_json(proposal, choice)
    test = write_run(tmp_path, 'test', 'test')
    monkeypatch.setattr('core.graded_evaluation.read_grades', Mock(side_effect=AssertionError('grades inspected')))
    with pytest.raises(ValueError, match='source dev'):
        offline_report(test, 'test', frozen_proposal=copied)


def test_test_access_cannot_switch_to_another_dev_proposal(tmp_path, monkeypatch):
    from core.graded_evaluation import offline_report
    dev = write_run(tmp_path, 'dev', 'dev')
    other = write_run(tmp_path, 'other-dev', 'dev')
    offline_report(dev, 'dev', True)
    offline_report(other, 'dev', True)
    test = write_run(tmp_path, 'test', 'test', grade='')
    offline_report(test, 'test', frozen_proposal=dev.with_suffix('.run')/'frozen-fit.json')
    monkeypatch.setattr('core.graded_evaluation.read_grades', Mock(side_effect=AssertionError('grades inspected')))
    with pytest.raises(ValueError, match='different frozen proposal'):
        offline_report(test, 'test', frozen_proposal=other.with_suffix('.run')/'frozen-fit.json')
