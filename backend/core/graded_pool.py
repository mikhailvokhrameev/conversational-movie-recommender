"""Run-scoped retrieval evidence and assessor-blind, reusable quality grades."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

from django.conf import settings

SCHEMA_VERSION = 1
RUBRIC_VERSION = 1
COLUMNS = ('query_id', 'query', 'split', 'tmdb_id', 'title', 'year', 'overview', 'grade')


def json_value(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    raise TypeError(f'unsupported snapshot value: {type(value).__name__}')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=json_value).encode()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_value))
    os.replace(temporary, path)


class RunStore:
    def __init__(self, csv_path, identity, resume=False):
        self.csv_path = Path(csv_path)
        self.directory = self.csv_path.with_suffix('.run')
        self.manifest_path = self.directory / 'manifest.json'
        if self.manifest_path.exists():
            if not resume:
                raise ValueError('export already exists; use --resume')
            self.manifest = json.loads(self.manifest_path.read_text())
            if self.manifest['identity'] != identity:
                raise ValueError('run manifest identity mismatch')
        else:
            if resume or self.csv_path.exists() or self.directory.exists():
                raise ValueError('export exists or resume manifest missing')
            self.directory.mkdir(parents=True)
            self.manifest = {'identity': identity, 'status': 'running', 'failures': {}}
            atomic_json(self.manifest_path, self.manifest)

    def path(self, query_id):
        return self.directory / (digest(query_id) + '.json')

    def load(self, query_id):
        path = self.path(query_id)
        if path.exists():
            value = json.loads(path.read_text())
            if value.get('complete'):
                return value
        return None

    def save(self, query_id, snapshot):
        atomic_json(self.path(query_id), snapshot)

    def fail(self, query_id, failures):
        self.manifest['status'] = 'failed'
        self.manifest['failures'][query_id] = failures
        atomic_json(self.manifest_path, self.manifest)


def identity(row):
    return (str(row['query_id']), row['query'], digest({'context': row.get('context', []), 'history': row.get('history')}), row['split'],
            str(row['tmdb_id']), row['rubric_version'])


def reuse_grades(rows, sources):
    grades = {}
    for row in [*sources, *rows]:
        grade = str(row.get('grade', ''))
        if not grade:
            continue
        if grade not in ('0', '1', '2', '3'):
            raise ValueError('grade must be blank or 0–3')
        key = identity(row)
        if key in grades and grades[key] != grade:
            raise ValueError('conflicting grades for query/movie identity')
        grades[key] = grade
    return [{**row, 'grade': grades.get(identity(row), '')} for row in rows]


def shuffled_union(groups, seed):
    result = list(dict.fromkeys(item for group in groups for item in group))
    random.Random(seed).shuffle(result)
    return result


def rerank_union(query, candidates, adapter, batch_size):
    from core.reranking import _movie_text
    from core.inference_diagnostics import inference_operation, require_healthy
    scores = {}
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]
        with inference_operation('reranking', 'pool_export'):
            values = adapter.predict_scores(query, [_movie_text(c) for c in batch])
            if len(values) != len(batch) or any(not math.isfinite(v) for v in values):
                raise ValueError('invalid reranker response')
        require_healthy()
        scores.update((str(c['id']), float(v)) for c, v in zip(batch, values))
    return scores


def replay(candidates, embedding, intent, raw_rerank, weights, pure=False):
    from core.scoring import score_candidates, normalize_scores, mmr_diversify
    from core.pipeline import apply_relevance_gate, exclude_candidates
    scored = score_candidates(candidates, embedding, intent, weights=weights)
    head = sorted(scored, key=lambda c: c['total'], reverse=True)[:settings.RERANK_TOP_K]
    norms = normalize_scores([raw_rerank[str(c['id'])] for c in head])
    totals = normalize_scores([c['total'] for c in head])
    weight = 1.0 if pure else settings.RERANK_WEIGHT
    ranked = [{**c, 'total': weight * norms[i] + (1-weight) * totals[i],
               'rerank_score': raw_rerank[str(c['id'])]} for i, c in enumerate(head)]
    ranked.sort(key=lambda c: c['total'], reverse=True)
    eligible = exclude_candidates(ranked, intent.get('reference_films'), None)
    return mmr_diversify(apply_relevance_gate(eligible, settings.RERANK_MIN_SCORE),
                         top_n=settings.EVAL_POOL_EXPORT['depth']), head


def capture_query(entry, prepared):
    from core.candidate_generation import generate_candidates
    from core.inference_backends import get_inference_backends
    from core.catalog_bounds import catalog_mean_rating
    from core.scoring import score_candidates
    started = time.monotonic()
    config = settings.EVAL_POOL_EXPORT
    variants = {}
    for name, quality in [('configured', settings.RRF_WEIGHTS['quality']), ('quality_1', config['quality_rrf_weight'])]:
        if prepared.intent.get('wants_obscure') and name == 'quality_1':
            continue
        weights = {**settings.RRF_WEIGHTS, 'quality': quality}
        variants[name] = generate_candidates(prepared.query_embedding, prepared.intent,
                                             query_text=prepared.query_text, rrf_weights=weights)
    union = {c['id']: c for candidates in variants.values() for c in candidates}
    scores = rerank_union(prepared.query_text, list(union.values()),
                          get_inference_backends().reranker, config['rerank_batch_size'])
    shipped = settings.SCORE_WEIGHTS
    rest = sum(v for k, v in shipped.items() if k != 'quality')
    quality_weight = config['quality_scoring_weight']
    quality_weights = {k: (quality_weight if k == 'quality' else v * (1-quality_weight)/rest)
                       for k, v in shipped.items()}
    groups, recall, replay_orders = [], {}, {}
    community = set(entry.get('community_recs', []))
    for name, candidates in variants.items():
        heads = {}
        replay_orders[name] = {}
        for blend, weights, pure in [('shipped', shipped, False), ('quality_03', quality_weights, False), ('reranker', shipped, True)]:
            top, head = replay(candidates, prepared.query_embedding, prepared.intent, scores, weights, pure)
            replay_orders[name][blend] = {'head': [c['id'] for c in head],
                                          'top': [c['id'] for c in top]}
            groups.append([c['id'] for c in top])
            heads[blend] = sorted(community & {c['tmdb_id'] for c in head})
        recall[name] = {'retrieved': sorted(community & {c['tmdb_id'] for c in candidates}),
                        'rerank_head': heads, 'community_count': len(community),
                        'retrieval_recall': (len(community & {c['tmdb_id'] for c in candidates}) / len(community)
                                             if community else None),
                        'rerank_head_recall': {blend: len(hits) / len(community) if community else None
                                              for blend, hits in heads.items()}}
    groups.append([c['id'] for c in union.values() if c['tmdb_id'] in community])
    selected = shuffled_union(groups, config['seed'])
    return {'complete': True, 'entry': entry, 'intent': prepared.intent,
            'candidate_count': len(union), 'duration_seconds': time.monotonic() - started,
            'rewritten_query': prepared.query_text, 'semantic_query': prepared.semantic_query,
            'query_embedding': prepared.query_embedding, 'variants': variants,
            'raw_features': score_candidates(list(union.values()), prepared.query_embedding, prepared.intent),
            'catalog_mean': catalog_mean_rating(), 'raw_reranker': scores,
            'community_recall': recall, 'replay_orders': replay_orders,
            'scoring_weights': {'shipped': shipped, 'quality_03': quality_weights},
            'selected': selected}


def rows_from_snapshots(snapshots):
    rows = []
    for snapshot in snapshots:
        entry = snapshot['entry']
        candidates = {c['id']: c for values in snapshot['variants'].values() for c in values}
        for movie_id in snapshot['selected']:
            c = candidates[movie_id]
            rows.append({'query_id': entry['id'], 'query': entry['query'], 'split': entry['split'],
                         'context': entry.get('context', []), 'history': entry.get('history'), 'rubric_version': RUBRIC_VERSION,
                         'tmdb_id': c['tmdb_id'], 'title': c['serial_name'],
                         'year': (c.get('release_date') or '')[:4], 'overview': c.get('description', ''), 'grade': ''})
    return rows


def read_grade_source(path):
    path = Path(path)
    with path.open(newline='') as stream:
        columns = csv.DictReader(stream).fieldnames or []
    if 'label' in columns and 'grade' not in columns:
        return []
    manifest = json.loads(path.with_suffix('.run').joinpath('manifest.json').read_text())
    entries = {str(e['id']): e for e in manifest['identity']['queries']}
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if not set(COLUMNS) <= set(reader.fieldnames or []):
            raise ValueError('grade source must be a graded pool CSV')
        rows = list(reader)
    for row in rows:
        entry = entries.get(row['query_id'])
        if entry is None or row['query'] != entry['query'] or row['split'] != entry['split']:
            raise ValueError('grade source query identity mismatch')
        row.update(context=entry.get('context', []), history=entry.get('history'), rubric_version=manifest['identity']['rubric_version'])
    return rows


def read_binary_source(path):
    """Legacy labels are binary evidence only, never quality grades."""
    path = Path(path)
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if any('label' not in row or row.get('label', '') not in ('', '0', '1') for row in rows):
        raise ValueError('invalid legacy binary label')
    return {'source': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'kind': 'binary_relevance', 'rows': rows}
