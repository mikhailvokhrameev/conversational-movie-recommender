"""CPU-only ranking replay from immutable, mode-scoped pool evidence.

No ORM, adapter factory or inference calls belong in this module. Unjudged
items are missing evidence, never negative labels. Fitting returns a proposal;
it does not change production configuration.
"""
import copy
import csv
import math
import random
from pathlib import Path

import numpy as np

from core.scoring import (SIGNALS, effective_weights, weighted_rating, mmr_diversify, normalize_scores,
                          _semantic_score, _metadata_score, _popularity_score)


def replay_ranking(snapshot: dict, params: dict, k: int) -> list[dict]:
    """Recompute full-pool scoring, head blending, gate, exclusions and MMR."""
    from core.pipeline import apply_relevance_gate, exclude_candidates
    if k < 1:
        raise ValueError('k must be positive')
    variant = params.get('_variant', 'configured')
    if snapshot['intent'].get('wants_obscure'):
        variant = 'configured'
    if variant not in snapshot['variants']:
        raise ValueError(f'missing saved retrieval variant: {variant}; export a new pool')
    candidates = snapshot['variants'][variant]
    scoring, rerank = params['scoring'], params['reranking']
    intent, mean = snapshot['intent'], snapshot['catalog_mean']
    weights = effective_weights(scoring['weights'], intent)
    raw = {
        'semantic': [_semantic_score(c, snapshot['query_embedding']) for c in candidates],
        'metadata': [_metadata_score(c, intent) for c in candidates],
        'popularity': [_popularity_score(c) for c in candidates],
        'quality': [weighted_rating(c.get('vote_average') or 0., c.get('vote_count') or 0,
                                    scoring['quality']['prior_votes'], mean) if mean is not None else 0.
                    for c in candidates],
    }
    norm = {s: normalize_scores(v, scoring['neutral_score']) for s, v in raw.items()}
    scored = [{**c, **{s: norm[s][i] for s in SIGNALS},
               'total': sum(norm[s][i]*weights[s] for s in SIGNALS),
               'raw_scores': {s: raw[s][i] for s in SIGNALS}}
              for i, c in enumerate(candidates)]
    if rerank['enabled'] and snapshot.get('rewritten_query', snapshot['entry']['query']):
        scored = sorted(scored, key=lambda c: c['total'], reverse=True)[:rerank['top_k']]
        if len(scored) >= 2:
            raw_scores = [snapshot['raw_reranker'][str(c['id'])] for c in scored]
            rn = normalize_scores(raw_scores, scoring['neutral_score'])
            sn = normalize_scores([c['total'] for c in scored], scoring['neutral_score'])
            w = rerank['weight']
            scored = [{**c, 'total': w*rn[i]+(1-w)*sn[i], 'rerank_score': raw_scores[i],
                              'pre_rerank_total': c['total']}
                      for i, c in enumerate(scored)]
            scored.sort(key=lambda c: c['total'], reverse=True)
    eligible = exclude_candidates(scored, intent.get('reference_films'),
                                  set(snapshot.get('exclude_ids', [])))
    gated = apply_relevance_gate(eligible, rerank['min_score'])
    return mmr_diversify(gated, top_n=k, lambda_param=params['diversification']['lambda'])


def read_grades(path, snapshots, split=None, reject_heldout_grades=False, supplemental_paths=()):
    from core.graded_pool import validated_grade_rows
    entries = {str(s['entry']['id']): s['entry'] for s in snapshots}
    grades = {}
    for location, row in validated_grade_rows([path, *supplemental_paths], snapshots, include_locations=True):
        entry = entries[row['query_id']]
        if split is not None and entry['split'] != split:
            if reject_heldout_grades and entry['split'] == 'test' and row['grade']:
                raise ValueError(f'{location}: dev fitting cannot consume a CSV with held-out test grades; use separate runs')
            continue
        if row['grade']:
            grades[(row['query_id'], row['tmdb_id'])] = int(row['grade'])
    return grades


def grade_report(snapshots: list[dict], grades: dict, params: dict, k: int) -> dict:
    records, missing, skipped = [], {}, 0
    for s in snapshots:
        q = str(s['entry']['id'])
        known = {str(c['tmdb_id']) for cs in s['variants'].values() for c in cs}
        judged = {m: g for (qid, m), g in grades.items() if qid == q and m in known}
        top = replay_ranking(s, params, k)
        for c in top[:5]:
            key = (q, str(c['tmdb_id']))
            if key not in grades:
                missing[key] = dict(query_id=q, query=s['entry']['query'], split=s['entry']['split'],
                                    tmdb_id=c['tmdb_id'], title=c['serial_name'],
                                    year=(c.get('release_date') or '')[:4],
                                    overview=c.get('description', ''), grade='')
        if len(judged) < 2:
            skipped += 1
            continue
        ids = [str(c['tmdb_id']) for c in top[:5]]
        complete = all(m in judged for m in ids)
        values = [judged[m] for m in ids if m in judged]
        ideal = sorted(judged.values(), reverse=True)[:5]
        dcg = lambda gs: sum((2**g-1)/math.log2(i+2) for i, g in enumerate(gs))
        community = {str(x) for x in s['entry'].get('community_recs', [])}
        records.append(dict(query_id=q, obscure=bool(s['intent'].get('wants_obscure')),
                            **{'ndcg@5': (dcg(values)/dcg(ideal) if dcg(ideal) else 0.) if complete else None,
                               'junk@5': sum(g == 0 for g in values)/5 if complete else None,
                               'good@5': sum(g >= 2 for g in values)/5 if complete else None,
                               'community_recall': len(community & set(ids))/len(community) if community else None,
                               'empty_result_rate': float(not top),
                               'judged_coverage': len(values)/len(ids) if ids else 1.}))
    def summarize(rows):
        result = {'query_count': len(rows), 'evaluated_query_count': len(rows)}
        for metric in ('ndcg@5', 'junk@5', 'good@5', 'community_recall', 'empty_result_rate', 'judged_coverage'):
            vals = [r[metric] for r in rows if r[metric] is not None]
            result[metric] = float(np.mean(vals)) if vals else None
        return result
    ordinary = summarize([r for r in records if not r['obscure']])
    obscure = summarize([r for r in records if r['obscure']])
    ordinary['query_count'] = sum(not s['intent'].get('wants_obscure', False) for s in snapshots)
    obscure['query_count'] = len(snapshots)-ordinary['query_count']
    overall = summarize(records)
    overall['query_count'] = len(snapshots)
    return dict(ordinary=ordinary, obscure=obscure, overall=overall,
                skipped_queries=skipped, complete_coverage=not missing,
                missing_pairs=list(missing.values()), per_query=records,
                metric_definitions={'ndcg@5': 'gain 2**grade - 1; ideal from judged saved pool',
                                    'junk@5': 'grade-0 count divided by 5',
                                    'good@5': 'grade>=2 count divided by 5',
                                    'judged_coverage': 'judged results divided by returned results'},
                pool_bias='Metrics are conditional on saved candidate pools and human-judged union; not catalog-wide recall.')



def paired_test_bootstrap(baseline, selected, samples):
    """Paired query resampling of the final frozen test choice, never tuning."""
    metrics = ('ndcg@5', 'junk@5', 'good@5', 'community_recall',
               'empty_result_rate', 'judged_coverage')
    base = {r['query_id']: r for r in baseline['per_query']}
    chosen = {r['query_id']: r for r in selected['per_query']}
    eligible = sorted(q for q in base.keys() & chosen.keys()
                      if base[q]['ndcg@5'] is not None and chosen[q]['ndcg@5'] is not None)
    complete = (baseline['complete_coverage'] and selected['complete_coverage']
                and not baseline['skipped_queries'] and not selected['skipped_queries'])
    result = dict(available=bool(eligible) and complete, seed=7, bootstrap_samples=samples,
                  confidence=.95, query_count=len(eligible), total_query_count=baseline['overall']['query_count'],
                  delta_direction='selected minus baseline',
                  scope='Separate held-out test: frozen selected configuration including its dev-selected gate versus recorded baseline',
                  metrics={})
    if not result['available']:
        result['reason'] = 'Complete judged comparison with at least one eligible paired query required; no test interval reported.'
        return result
    for metric in metrics:
        ids = [q for q in eligible if base[q][metric] is not None and chosen[q][metric] is not None]
        if not ids:
            result['metrics'][metric] = dict(query_count=0, delta_mean=None, delta_95=None)
            continue
        delta = np.array([chosen[q][metric]-base[q][metric] for q in ids])
        rng = np.random.default_rng(7)
        means = np.mean(rng.choice(delta, (samples, len(delta))), axis=1)
        result['metrics'][metric] = dict(query_count=len(ids), delta_mean=float(np.mean(delta)),
                                         delta_95=np.quantile(means, [.025, .975]).tolist())
    return result

def grouped_folds(snapshots):
    ids = sorted({str(s['entry']['id']) for s in snapshots})
    if len(ids) < 5:
        raise ValueError('Need at least five eligible dev queries with two grades; grade more queries.')
    random.Random(7).shuffle(ids)
    return [ids[i::5] for i in range(5)]


def _grid(config):
    baseline = copy.deepcopy(config)
    if config['evaluation']['pool_export']['quality_rrf_weight'] != 1.:
        raise ValueError('fitting needs saved quality RRF 1.0; export a new pool with that variant')
    result = []
    weights = baseline['scoring']['weights']
    for quality in dict.fromkeys([weights['quality'], *config['evaluation']['grade_fit']['quality_weights']]):
        rest = sum(v for s, v in weights.items() if s != 'quality')
        if rest <= 0:
            raise ValueError('baseline non-quality weights must have positive mass')
        for rrf in config['evaluation']['grade_fit']['quality_rrf_weights']:
            p = copy.deepcopy(baseline)
            p['scoring']['weights'] = {s: quality if s == 'quality' else v*(1-quality)/rest
                                       for s, v in weights.items()}
            base_rrf = baseline['retrieval']['rrf_weights']['quality']
            if base_rrf not in (0., 1.):
                raise ValueError('fitting requires a recorded baseline quality RRF of 0 or 1')
            p['_variant'] = 'configured' if rrf == base_rrf else f'quality_{int(rrf)}'
            p['retrieval']['rrf_weights']['quality'] = rrf
            result.append(p)
    return result


def _select(reports, configs, baseline):
    def key(i):
        weights = configs[i]['scoring']['weights']
        distance = sum(abs(v-baseline['scoring']['weights'][s]) for s, v in weights.items())
        distance += abs(configs[i]['retrieval']['rrf_weights']['quality']-
                        baseline['retrieval']['rrf_weights']['quality'])
        return (-reports[i]['overall']['ndcg@5'], reports[i]['overall']['junk@5'], distance, i)
    return min(range(len(configs)), key=key)


def calibrate_gate(snapshots, grades, params):
    """Mode-specific raw score table; never transfer a BGE threshold to Voyage."""
    p = copy.deepcopy(params)
    p['reranking']['min_score'] = None
    ungated = [replay_ranking(s, p, p['reranking']['top_k']) for s in snapshots]
    relevant = []
    full = True
    missing = {}
    answerable = []
    for s, cs in zip(snapshots, ungated):
        q = str(s['entry']['id'])
        known = {str(c['tmdb_id']) for candidates in s['variants'].values() for c in candidates}
        answerable.append(any(g >= 2 for (qid, m), g in grades.items() if qid == q and m in known))
        for c in cs:
            if (q, str(c['tmdb_id'])) not in grades:
                missing[(q, str(c['tmdb_id']))] = dict(query_id=q, query=s['entry']['query'],
                    split=s['entry']['split'], tmdb_id=c['tmdb_id'], title=c['serial_name'],
                    year=(c.get('release_date') or '')[:4], overview=c.get('description', ''), grade='')
        full &= all((q, str(c['tmdb_id'])) in grades for c in cs)
        relevant.append([c for c in cs if grades.get((q, str(c['tmdb_id'])), -1) >= 2])
    thresholds = [None] + sorted({c['rerank_score'] for cs in ungated for c in cs if 'rerank_score' in c})
    table = []
    total = sum(map(len, relevant))
    for t in thresholds:
        kept = [sum(t is None or c.get('rerank_score', -math.inf) >= t for c in cs) for cs in relevant]
        loss = (total-sum(kept))/total if total else None
        table.append(dict(threshold=t, relevant_retained=sum(kept), relevant_total=total,
                          dropped_fraction=loss, answerable_queries=sum(answerable),
                          answerable_retained=sum(bool(n) for n in kept),
                          valid=bool(total) and loss <= .05 and all(n for yes, n in zip(answerable, kept) if yes)))
    valid = [r for r in table if r['valid']]
    calibrated = full and bool(valid) and params['reranking']['enabled']
    return dict(calibrated=calibrated, complete_coverage=full,
                threshold=valid[-1]['threshold'] if calibrated else None, table=table,
                missing_pairs=list(missing.values()), judged_coverage=(sum(map(len, ungated))-len(missing))/sum(map(len, ungated)) if any(ungated) else 1.,
                reason=None if calibrated else 'Full judged head and relevant dev evidence required.')


def fit_ranking_params(snapshots: list[dict], grades: dict, config: dict) -> dict:
    if any(s['entry']['split'] != 'dev' for s in snapshots):
        raise ValueError('fit is dev only; freeze choices before reading test grades')
    eligible = [s for s in snapshots if sum((str(s['entry']['id']), str(c['tmdb_id'])) in grades
                for c in {c['tmdb_id']: c for cs in s['variants'].values() for c in cs}.values()) >= 2]
    folds = grouped_folds(eligible)
    configs = _grid(config)
    k = config['diversification']['top_n']
    reports = [grade_report(eligible, grades, p, k) for p in configs]
    missing = {(r['query_id'], str(r['tmdb_id'])): r for report in reports for r in report['missing_pairs']}
    base = grade_report(eligible, grades, config, k)
    if missing:
        return dict(applied=False, configurations_evaluated=len(configs), complete_coverage=False,
                    missing_pairs=list(missing.values()), baseline=base,
                    reason='Grade all compared top-5 pairs before fitting or acceptance comparisons.')
    fold_rows = []
    paired_base, paired_selected = [], []
    for fold in folds:
        train = [s for s in eligible if str(s['entry']['id']) not in fold]
        test = [s for s in eligible if str(s['entry']['id']) in fold]
        train_reports = [grade_report(train, grades, p, k) for p in configs]
        choice = _select(train_reports, configs, config)
        b = grade_report(test, grades, config, k)
        selected = grade_report(test, grades, configs[choice], k)
        fold_rows.append(dict(query_ids=fold, selected_index=choice,
                              baseline=b['overall']['ndcg@5'], selected=selected['overall']['ndcg@5']))
        bd = {r['query_id']: r['ndcg@5'] for r in b['per_query']}
        for r in selected['per_query']:
            paired_base.append(bd[r['query_id']]); paired_selected.append(r['ndcg@5'])
    delta = np.array(paired_selected)-np.array(paired_base)
    rng = np.random.default_rng(7)
    bootstrap = np.mean(rng.choice(delta, (config['evaluation']['grade_fit']['bootstrap_samples'], len(delta))), axis=1)
    choice = _select(reports, configs, config)
    selected = configs[choice]
    gate = calibrate_gate(eligible, grades, selected)
    if gate['calibrated']:
        selected['reranking']['min_score'] = gate['threshold']
    selected_report = grade_report(eligible, grades, selected, k)
    summary = {name: dict(mean=float(np.mean([r[name] for r in fold_rows])),
                          std=float(np.std([r[name] for r in fold_rows]))) for name in ('baseline', 'selected')}
    return dict(applied=True, complete_coverage=True, missing_pairs=[], configurations_evaluated=len(configs),
                skipped_queries=len(snapshots)-len(eligible), selected_params=selected, baseline=base, selected=selected_report, gate=gate,
                cross_validation=dict(folds=fold_rows, **summary, seed=7,
                                      scope='Held-out ranking-weight selection at the recorded baseline gate; excludes subsequent final dev gate calibration',
                                      paired_query_bootstrap_delta_95=np.quantile(bootstrap, [.025, .975]).tolist()),
                pool_bias=base['pool_bias'], scope='dev only; freeze this proposal before test evaluation')


def _proposal_compatibility(identity):
    """Compatibility intentionally excludes query lists and split assignment."""
    from core.graded_pool import digest
    required = ('catalog_sha256', 'index', 'baseline_params', 'inference')
    if any(key not in identity or identity[key] is None for key in required):
        raise ValueError('incompatible proposal: missing catalog/index/baseline provenance; export a new pool')
    inference = identity['inference']
    if not inference.get('mode') or not inference.get('components'):
        raise ValueError('incompatible proposal: missing mode/provider/model provenance')
    return {'mode': inference['mode'], 'components': inference['components'],
            'catalog_sha256': identity['catalog_sha256'], 'index': identity['index'],
            'baseline_sha256': digest(identity['baseline_params'])}


def _load_frozen_proposal(proposal_path, identity):
    import json
    from core.graded_pool import digest
    frozen = json.loads(proposal_path.read_text())
    if frozen.get('compatibility') != _proposal_compatibility(identity):
        raise ValueError('incompatible frozen proposal: mode/provider, catalog/index provenance or baseline differs')
    source_manifest = Path(frozen['source_dev_manifest'])
    source = json.loads(source_manifest.read_text())
    if digest(source['identity']) != frozen['source_dev_identity_sha256']:
        raise ValueError('incompatible proposal: source dev identity has changed')
    original = json.loads((source_manifest.parent/'frozen-fit.json').read_text())
    if digest(original) != digest(frozen):
        raise ValueError('incompatible proposal: source dev choice has been replaced')
    trained_ids = {str(entry['id']) for entry in source['identity']['queries']
                   if entry['split'] == 'dev'}
    heldout_ids = {str(entry['id']) for entry in identity['queries']
                   if entry['split'] == 'test'}
    overlap = trained_ids & heldout_ids
    if overlap:
        raise ValueError(f"source dev and target test query IDs overlap: {sorted(overlap)}; export a disjoint held-out split")
    return frozen, source_manifest.parent


def offline_report(path, split='dev', fit=False, frozen_proposal=None):
    """Replay a frozen dev proposal against a compatible independent test run.

    First test access locks its source dev run before reading test grades,
    including attempts that later fail validation or have incomplete coverage.
    """
    import json
    from core.graded_pool import atomic_json, digest, COLUMNS
    path = Path(path)
    if split not in ('dev', 'test'):
        raise ValueError('grade report split must be dev or test')
    if fit and split == 'test':
        raise ValueError('cannot fit test grades')
    if fit and frozen_proposal:
        raise ValueError('cannot combine fitting with a frozen proposal')
    directory = path.with_suffix('.run')
    fit_path, test_path = directory/'frozen-fit.json', directory/'test-report.json'
    proposal_path = Path(frozen_proposal) if frozen_proposal else fit_path
    if split == 'test' and not proposal_path.is_file():
        raise ValueError('test reporting requires a frozen proposal; fit a separate dev run first and pass --frozen-proposal PATH')
    if fit and (directory/'test-access.json').exists():
        raise ValueError('test access already recorded; the frozen dev choice cannot be replaced')
    manifest_path = directory/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['status'] != 'complete':
        raise ValueError('pool export is incomplete; resume it before offline evaluation')
    identity = manifest['identity']
    if 'baseline_params' not in identity:
        raise ValueError('manifest lacks baseline_params; export a new pool for faithful offline replay')
    baseline = identity['baseline_params']
    mode = identity['inference']['mode']
    frozen = None
    if not fit and proposal_path.exists():
        frozen, source_directory = _load_frozen_proposal(proposal_path, identity)
        if split == 'test':
            # Durably lock both the source dev choice and target test run before
            # reading even one held-out grade or calculating any test metric.
            access = {'proposal_sha256': digest(frozen),
                      'source_dev_identity_sha256': frozen['source_dev_identity_sha256']}
            for lock_directory in dict.fromkeys([source_directory, directory]):
                lock_path = lock_directory/'test-access.json'
                if lock_path.exists():
                    if json.loads(lock_path.read_text()) != access:
                        raise ValueError('test access already bound to a different frozen proposal')
                else:
                    atomic_json(lock_path, access)
    snapshots = []
    for entry in identity['queries']:
        item = json.loads((directory/(digest(entry['id'])+'.json')).read_text())
        if not item.get('complete') or item['entry'] != entry:
            raise ValueError('incomplete or mismatched query snapshot; resume pool export')
        snapshots.append(item)
    worksheet = path.with_suffix('.missing.csv')
    grades = read_grades(path, snapshots, split=split, reject_heldout_grades=fit,
                         supplemental_paths=[worksheet] if worksheet.exists() else [])
    selected_snapshots = [item for item in snapshots if item['entry']['split'] == split]
    if not selected_snapshots:
        raise ValueError(f'no {split} snapshots; export that split first')
    k = baseline['diversification']['top_n']
    report = dict(mode=mode, split=split, baseline=grade_report(selected_snapshots, grades, baseline, k))
    if fit:
        compatibility = _proposal_compatibility(identity)
        result = fit_ranking_params(selected_snapshots, grades, baseline)
        report['fit'] = result
        if result['applied']:
            proposal = copy.deepcopy(result['selected_params'])
            if result['gate']['calibrated']:
                proposal['reranking']['min_score'] = result['gate']['threshold']
            atomic_json(fit_path, dict(source_dev_identity_sha256=digest(identity),
                                      source_dev_manifest=str(manifest_path.resolve()),
                                      compatibility=compatibility, params=proposal))
    elif frozen:
        report['frozen_choice'] = True
        report['source_dev_identity_sha256'] = frozen['source_dev_identity_sha256']
        report['selected'] = grade_report(selected_snapshots, grades, frozen['params'], k)
    else:
        report['frozen_choice'] = False
    compared = [report['baseline']]
    if 'selected' in report:
        compared.append(report['selected'])
    if fit:
        compared.append(report['fit'])
        if report['fit'].get('gate'):
            compared.append(report['fit']['gate'])
    missing = {(r['query_id'], str(r['tmdb_id'])): r for item in compared for r in item.get('missing_pairs', [])}
    report['comparison_complete'] = (all(item.get('complete_coverage', False) for item in compared)
                                     and not any(item.get('skipped_queries', 0) for item in compared))
    if missing:
        # Retain grades already entered and append new pairs without duplicates.
        with path.open(newline='') as stream:
            old = list(csv.DictReader(stream))
        if worksheet.exists():
            with worksheet.open(newline='') as stream:
                old.extend(csv.DictReader(stream))
        rows = {}
        for row in old:
            key = (row['query_id'], row['tmdb_id'])
            if key not in rows or row.get('grade', '').strip():
                rows[key] = row
        for key, row in missing.items():
            rows.setdefault(key, row)
        for key, grade in grades.items():
            if key in rows:
                rows[key]['grade'] = str(grade)
        destination = worksheet
        temporary = destination.with_suffix('.csv.tmp')
        with temporary.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS, extrasaction='ignore')
            writer.writeheader(); writer.writerows(rows.values())
        temporary.replace(destination)
        report['missing_grades_csv'] = str(destination)
    if split == 'test' and report['frozen_choice']:
        report['paired_query_bootstrap'] = paired_test_bootstrap(
            report['baseline'], report['selected'], baseline['evaluation']['grade_fit']['bootstrap_samples'])
    if split == 'test' and report['comparison_complete'] and report['frozen_choice']:
        if test_path.exists():
            previous = json.loads(test_path.read_text())
            if previous != report:
                raise ValueError('completed test report is frozen; it is not a new tuning opportunity')
        else:
            atomic_json(test_path, report)
    return report
