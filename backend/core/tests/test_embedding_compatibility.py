"""Cross-backend index checks, entirely offline (no DB or model weights)."""
from copy import deepcopy
from importlib import import_module
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from django.core.management.base import CommandError
from django.db import migrations

from core.embedding_provenance import ProvenanceError, compare_provenance, current_provenance, plan_generate
from core.inference_config import resolve_inference_config
from movies.management.commands.check_embedding_index import Command
from movies.models import EmbeddingIndex

CHECK = 'movies.management.commands.check_embedding_index'
GENERATE = 'movies.management.commands.generate_embeddings'


def origin(provider='local'):
    return {**current_provenance(), 'provider': provider,
            'api_provider': 'deepinfra' if provider == 'openrouter' else None,
            'revision': None if provider == 'openrouter' else 'local-revision'}


def test_remote_origin_keeps_canonical_model_and_unknown_revision(settings):
    settings.EFFECTIVE_PARAMS = resolve_inference_config(deepcopy(settings.PARAMS), 'openrouter')
    provenance = current_provenance()
    assert provenance['model_name'] == 'BAAI/bge-m3'
    assert provenance['provider'] == 'openrouter'
    assert provenance['api_provider'] == 'deepinfra'
    assert provenance['revision'] is None


@pytest.mark.parametrize('old,new', [('local', 'openrouter'), ('openrouter', 'local')])
def test_origin_change_requires_read_check_and_forbids_partial_generation(old, new):
    check = compare_provenance(origin(old), origin(new))
    assert check.status == 'backend_changed'
    assert check.origin_differences
    assert check.notes
    with pytest.raises(ProvenanceError, match='--all'):
        plan_generate(check, True, False, False)
    assert plan_generate(check, True, True, False) == 'embed_all'
    assert plan_generate(check, False, False, False) == 'embed_missing'


@pytest.mark.parametrize('changes', [dict(model_name='other'), dict(dimensions=768), dict(text_version=2), dict(text_config={})])
def test_hard_mismatch_wins_over_origin_change(changes):
    assert compare_provenance(origin(), {**origin('openrouter'), **changes}).status == 'mismatch'


def run_check(stored, current, rows, vectors):
    output = StringIO()
    with patch(f'{CHECK}.Movie') as movie, patch(f'{CHECK}.EmbeddingIndex') as index, \
         patch(f'{CHECK}.current_provenance', return_value=current), \
         patch(f'{CHECK}.encode_texts', return_value=vectors, create=True) as encode:
        index.stored.return_value = stored
        query = movie.objects.filter.return_value
        query.exists.return_value = bool(rows)
        query.order_by.return_value.values_list.return_value.__getitem__.return_value = rows
        Command(stdout=output).handle()
        index.record.assert_not_called()
        movie.objects.bulk_update.assert_not_called()
    return output.getvalue(), encode, query


def rows_for(count):
    return [(i, f'Film {i}', ['Drama'], 'plot', ['k'], [1., 0.]) for i in range(count)]


@pytest.mark.parametrize('old,new', [('local', 'openrouter'), ('openrouter', 'local')])
def test_small_index_checks_all_rows_in_order_without_changing_origin(old, new):
    output, encode, query = run_check(origin(old), origin(new), rows_for(2), [[1., 0.], [2., 0.]])
    encode.assert_called_once_with(['Film 0. Drama. plot. Keywords: k', 'Film 1. Drama. plot. Keywords: k'])
    query.order_by.assert_called_once_with('tmdb_id')
    assert query.order_by.return_value.values_list.return_value.__getitem__.call_args.args == (slice(None, 32),)
    assert 'compatib' in output.lower()
    assert 'revision' in output
    assert 'unchanged' in output.lower()


def test_origin_change_with_empty_index_never_loads_embedder():
    _, encode, _ = run_check(origin(), origin('openrouter'), [], [])
    encode.assert_not_called()


@pytest.mark.parametrize('vectors', [[[0.998, 0.064]], [[float('nan'), 0.]], [[0., 0.]], [], [[1., 0., 0.]]])
def test_bad_or_missing_sample_fails_with_original_mode_and_explicit_rebuild_guidance(vectors):
    with pytest.raises(CommandError, match='original.*mode|original.*backend') as exc:
        run_check(origin(), origin('openrouter'), rows_for(1), vectors)
    assert '--all' in str(exc.value)


def test_one_bad_sample_cannot_be_hidden_by_average_similarity():
    with pytest.raises(CommandError):
        run_check(origin(), origin('openrouter'), rows_for(2), [[1., 0.], [0., 1.]])


def test_api_provider_change_requires_check():
    old = origin('openrouter')
    assert compare_provenance(old, {**old, 'api_provider': 'other'}).status == 'backend_changed'


def test_embedding_index_round_trips_origin_without_database():
    with patch.object(EmbeddingIndex.objects, 'filter') as query:
        query.return_value.first.return_value = SimpleNamespace(**origin('openrouter'))
        assert EmbeddingIndex.stored() == origin('openrouter')


def test_migration_gives_legacy_rows_local_origin_without_inventing_api_provider():
    migration = import_module('movies.migrations.0010_embeddingindex_origin').Migration
    fields = {op.name: op.field for op in migration.operations if isinstance(op, migrations.AddField)}
    assert fields['provider'].get_default() == 'local'
    assert fields['api_provider'].null
    assert fields['api_provider'].get_default() is None
    assert migration.dependencies == [('movies', '0009_null_zero_runtime')]


def test_generate_refuses_changed_origin_even_after_successful_read_check():
    from movies.management.commands.generate_embeddings import Command as Generate
    stored, current = origin(), origin('openrouter')
    run_check(stored, current, rows_for(1), [[1., 0.]])
    with patch(f'{GENERATE}.Movie') as movie, patch(f'{GENERATE}.EmbeddingIndex') as index, \
         patch(f'{GENERATE}.current_provenance', return_value=current), \
         patch(f'{GENERATE}.encode_texts') as encode:
        index.stored.return_value = stored
        movie.objects.filter.return_value.exists.return_value = True
        with pytest.raises(CommandError, match='--all'):
            Generate(stdout=StringIO()).handle(batch_size=1, all=False, adopt=False)
        encode.assert_not_called()
        index.record.assert_not_called()


@pytest.mark.parametrize('size,threshold', [(0, .999), (33, .999), (True, .999), (32, .9989),
                                          (32, 1.1), (32, float('nan')), (32, True)])
def test_compatibility_settings_cannot_weaken_required_check(size, threshold):
    from recommender.settings import _validate_embedding_compatibility
    with pytest.raises(ValueError, match='embedding.compatibility'):
        _validate_embedding_compatibility({'compatibility': {'sample_size': size, 'min_cosine': threshold}})


def test_sample_size_and_threshold_come_from_yaml(settings):
    settings.EFFECTIVE_PARAMS = deepcopy(settings.EFFECTIVE_PARAMS)
    settings.EFFECTIVE_PARAMS['embedding']['compatibility'] = {'sample_size': 1, 'min_cosine': 1.0}
    _, _, query = run_check(origin(), origin('openrouter'), rows_for(1), [[1., 0.]])
    assert query.order_by.return_value.values_list.return_value.__getitem__.call_args.args == (slice(None, 1),)
    with pytest.raises(CommandError):
        run_check(origin(), origin('openrouter'), rows_for(1), [[1., .001]])


def test_minimum_cosine_boundary_is_inclusive():
    from core.embedding_provenance import validate_compatibility_vectors
    validate_compatibility_vectors([[1., 0.]], [[.999, (1 - .999 ** 2) ** .5]], .999)


def test_migration_applies_origin_fields_to_legacy_state_without_database():
    from django.db.migrations.loader import MigrationLoader
    from unittest.mock import MagicMock
    loader = MigrationLoader(None)
    state = loader.project_state([('movies', '0009_null_zero_runtime')])
    old_model = state.apps.get_model('movies', 'EmbeddingIndex')
    assert 'provider' not in {f.name for f in old_model._meta.fields}
    schema_editor = MagicMock()
    migration = import_module('movies.migrations.0010_embeddingindex_origin').Migration('0010_embeddingindex_origin', 'movies')
    new_state = migration.apply(state, schema_editor)
    model = new_state.apps.get_model('movies', 'EmbeddingIndex')
    assert model().provider == 'local'
    assert model().api_provider is None
    assert [call.args[1].name for call in schema_editor.add_field.call_args_list] == ['provider', 'api_provider']


def test_all_rebuild_buffers_before_one_atomic_update_and_records_remote_origin():
    import numpy as np
    from contextlib import contextmanager
    from movies.management.commands.generate_embeddings import Command as Generate
    events = []
    stored, current = origin(), origin('openrouter')
    movie_rows = [(1, 'First', [], 'plot', []), (2, 'Second', [], 'plot', [])]
    movie_objects = [SimpleNamespace(id=i, embedding=[0., 1.]) for i in (1, 2)]

    @contextmanager
    def atomic():
        events.append('begin')
        yield
        events.append('commit')

    def encode(texts):
        events.append('encode')
        return np.array([[1., 0.]])

    with patch(f'{GENERATE}.Movie') as movie, patch(f'{GENERATE}.EmbeddingIndex') as index, \
         patch(f'{GENERATE}.current_provenance', return_value=current), \
         patch(f'{GENERATE}.encode_texts', side_effect=encode), \
         patch(f'{GENERATE}.transaction.atomic', side_effect=atomic):
        index.stored.return_value = stored
        movie.objects.all.return_value.values_list.return_value = movie_rows
        movie.objects.filter.return_value.__iter__.return_value = iter(movie_objects)
        movie.objects.bulk_update.side_effect = lambda *a, **kw: events.append('update')
        index.record.side_effect = lambda *a: events.append('record')
        Generate(stdout=StringIO()).handle(batch_size=1, all=True, adopt=False)
        index.record.assert_called_once_with(current)
        assert [m.embedding for m in movie_objects] == [[1., 0.], [1., 0.]]
    assert events == ['encode', 'encode', 'begin', 'update', 'record', 'commit']


def test_failed_explicit_rebuild_leaves_old_vectors_and_origin():
    import numpy as np
    from movies.management.commands.generate_embeddings import Command as Generate
    movie_objects = [SimpleNamespace(id=i, embedding=[0., 1.]) for i in (1, 2)]
    with patch(f'{GENERATE}.Movie') as movie, patch(f'{GENERATE}.EmbeddingIndex') as index, \
         patch(f'{GENERATE}.current_provenance', return_value=origin('openrouter')), \
         patch(f'{GENERATE}.encode_texts', side_effect=[np.array([[1., 0.]]), RuntimeError('interrupted')]), \
         patch(f'{GENERATE}.transaction.atomic') as atomic:
        index.stored.return_value = origin()
        movie.objects.all.return_value.values_list.return_value = [(i, 'Film', [], '', []) for i in (1, 2)]
        movie.objects.filter.return_value.__iter__.return_value = iter(movie_objects)
        with pytest.raises(RuntimeError, match='interrupted'):
            Generate(stdout=StringIO()).handle(batch_size=1, all=True, adopt=False)
        atomic.assert_not_called()
        index.record.assert_not_called()
        movie.objects.bulk_update.assert_not_called()
        assert [m.embedding for m in movie_objects] == [[0., 1.], [0., 1.]]
