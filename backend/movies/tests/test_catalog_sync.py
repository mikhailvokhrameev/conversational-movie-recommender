"""Offline unit coverage; PostgreSQL publication coverage is marked django_db."""
from datetime import date
from unittest.mock import Mock

import pytest

TODAY = date(2026, 11, 7)


def row(**changes):
    data = dict(tmdb_id=1, serial_name='Example', original_title='', genres=[],
                country=[], original_language='en', keywords=[],
                release_date=date(2024, 1, 1), description='An overview', runtime=None,
                popularity=1., vote_average=7., vote_count=20,
                poster_path='/poster.jpg', adult=False, status='Released')
    data.update(changes)
    return data


def test_sync_interface_exists():
    from movies import catalog_sync
    assert callable(catalog_sync.sync_catalog)


@pytest.mark.parametrize('field,value', [('adult', True), ('status', 'Planned'),
    ('description', '  '), ('poster_path', None), ('vote_count', 19),
    ('release_date', date(2027, 1, 1)), ('release_date', date(2023, 12, 31))])
def test_eligibility(field, value):
    from movies.catalog_sync import _eligible
    data = row()
    data[field] = value
    assert not _eligible(data, TODAY)


@pytest.mark.parametrize('runtime', [None, 0])
def test_unknown_runtime_is_eligible(runtime):
    from movies.catalog_sync import _eligible
    assert _eligible(row(runtime=runtime), TODAY)


@pytest.fixture
def mocked_store(monkeypatch):
    from movies import catalog_sync as sync
    movies = Mock()
    movies.filter.return_value.values_list.return_value = []
    movies.filter.return_value.exists.return_value = False
    movies.exists.return_value = False
    monkeypatch.setattr(sync.Movie, 'objects', movies)
    monkeypatch.setattr(sync.EmbeddingIndex, 'stored', lambda: None)
    monkeypatch.setattr(sync.EmbeddingIndex, 'record', Mock())
    monkeypatch.setattr(sync, 'encode_texts', Mock(return_value=[[1.] * sync.settings.EMBEDDING_DIMENSIONS]))
    return sync, movies


def test_partial_traversal_never_touches_database(mocked_store):
    sync, movies = mocked_store
    def partial(*args):
        yield row()
        raise sync.TMDBSyncError('late failure')
    with pytest.raises(sync.TMDBSyncError):
        sync.sync_catalog(Mock(iter_movies=partial), today=TODAY)
    movies.filter.assert_not_called()
    sync.encode_texts.assert_not_called()


def test_csv_without_ready_index_skips_and_warns(mocked_store, caplog):
    sync, movies = mocked_store
    movies.exists.return_value = True
    result = sync.sync_catalog(Mock(iter_movies=Mock(return_value=iter([row()]))), today=TODAY)
    assert result.checked == 1 and result.added == 0
    assert 'index' in caplog.text.lower()
    sync.encode_texts.assert_not_called()
    movies.bulk_create.assert_not_called()


@pytest.mark.parametrize('vectors', [[], [[1.]], [[float('nan')] * 1024]])
def test_invalid_vectors_prevent_publication(mocked_store, vectors):
    sync, movies = mocked_store
    sync.encode_texts.return_value = vectors
    with pytest.raises(sync.TMDBSyncError):
        sync.sync_catalog(Mock(iter_movies=Mock(return_value=iter([row()]))), today=TODAY)
    movies.bulk_create.assert_not_called()


def test_embedder_failure_prevents_publication(mocked_store):
    sync, movies = mocked_store
    sync.encode_texts.side_effect = RuntimeError('private provider error')
    with pytest.raises(sync.TMDBSyncError, match='Catalog synchronization failed'):
        sync.sync_catalog(Mock(iter_movies=Mock(return_value=iter([row()]))), today=TODAY)
    movies.bulk_create.assert_not_called()


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch):
    import socket
    original = socket.socket.connect
    def local_only(sock, address):
        if isinstance(address, tuple) and address[0] not in {'127.0.0.1', 'localhost', '::1'}:
            raise AssertionError('External networking is forbidden in catalog sync tests')
        return original(sock, address)
    monkeypatch.setattr(socket.socket, 'connect', local_only)


def test_origin_change_prevents_embedding(mocked_store, monkeypatch):
    sync, movies = mocked_store
    current = sync.current_provenance()
    monkeypatch.setattr(sync.EmbeddingIndex, 'stored', lambda: dict(current, provider='another-provider'))
    movies.filter.return_value.exists.return_value = True
    # No missing vectors; an existing origin must reach plan_generate.
    def filtered(**kwargs):
        result = Mock()
        result.values_list.return_value = []
        result.exists.return_value = kwargs.get('embedding__isnull') is False
        return result
    movies.filter.side_effect = filtered
    with pytest.raises(sync.TMDBSyncError):
        sync.sync_catalog(Mock(iter_movies=Mock(return_value=iter([row()]))), today=TODAY)
    sync.encode_texts.assert_not_called()
    movies.bulk_create.assert_not_called()


@pytest.fixture
def publication(monkeypatch):
    from movies import catalog_sync as sync
    monkeypatch.setattr(sync, 'encode_texts', lambda texts: [[1.] * sync.settings.EMBEDDING_DIMENSIONS for _ in texts])
    return sync


@pytest.mark.django_db(transaction=True)
def test_repeat_threshold_existing_unchanged_and_targeted_index(publication):
    sync = publication
    data = row()
    data['vote_count'] = 19
    client = Mock(iter_movies=Mock(side_effect=lambda *a: iter([dict(data)])))
    assert sync.sync_catalog(client, today=TODAY).added == 0
    data['vote_count'] = 20
    assert sync.sync_catalog(client, today=TODAY).added == 1
    movie = sync.Movie.objects.get(tmdb_id=1)
    assert movie.runtime is None and movie.embedding is not None and movie.search_vector is not None
    data['serial_name'] = 'Changed upstream'
    assert sync.sync_catalog(client, today=TODAY).added == 0
    movie.refresh_from_db()
    assert movie.serial_name == 'Example'
    assert client.iter_movies.call_args.args == (date(2024, 1, 1), TODAY)


@pytest.mark.django_db(transaction=True)
def test_index_failure_rolls_back_all_rows_and_provenance(publication, monkeypatch):
    sync = publication
    monkeypatch.setattr(sync, 'refresh_search_vectors', Mock(side_effect=RuntimeError('index failure')))
    with pytest.raises(sync.TMDBSyncError):
        sync.sync_catalog(Mock(iter_movies=Mock(return_value=iter([row()]))), today=TODAY)
    assert not sync.Movie.objects.exists()
    assert sync.EmbeddingIndex.stored() is None


@pytest.mark.django_db(transaction=True)
def test_concurrent_syncs_publish_once(publication, monkeypatch):
    """Each worker prepares the same movie before contending for the DB lock."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from django.db import close_old_connections
    sync = publication
    barrier = Barrier(2)
    def encode(texts):
        barrier.wait(timeout=10)
        return [[1.] * sync.settings.EMBEDDING_DIMENSIONS for _ in texts]
    monkeypatch.setattr(sync, 'encode_texts', encode)
    def run():
        close_old_connections()
        try:
            return sync.sync_catalog(Mock(iter_movies=Mock(return_value=iter([row()]))), today=TODAY).added
        finally:
            close_old_connections()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: run(), range(2)))
    assert sorted(results) == [0, 1]
    assert sync.Movie.objects.count() == 1


@pytest.mark.django_db(transaction=True)
def test_provenance_changed_during_inference_prevents_publication(publication, monkeypatch):
    sync = publication
    current = sync.current_provenance()
    sync.EmbeddingIndex.record(current)
    def encode(texts):
        # Another writer changed the record during the unlocked inference phase.
        sync.EmbeddingIndex.record(dict(current, model_name='different-model'))
        return [[1.] * sync.settings.EMBEDDING_DIMENSIONS for _ in texts]
    monkeypatch.setattr(sync, 'encode_texts', encode)
    with pytest.raises(sync.TMDBSyncError):
        sync.sync_catalog(Mock(iter_movies=Mock(return_value=iter([row()]))), today=TODAY)
    assert not sync.Movie.objects.exists()
    assert sync.EmbeddingIndex.stored()['model_name'] == 'different-model'


def test_empty_catalog_refreshes_old_origin_and_prepares_outside_transaction(mocked_store, monkeypatch):
    from contextlib import contextmanager
    sync, movies = mocked_store
    state = {'atomic': False, 'traversed': False, 'encoded': False}
    current = sync.current_provenance()
    old = dict(current, provider='old-provider', api_provider='old-api')
    monkeypatch.setattr(sync.EmbeddingIndex, 'stored', lambda: old)
    @contextmanager
    def atomic():
        assert state['traversed'] and state['encoded']
        state['atomic'] = True
        yield
        state['atomic'] = False
    monkeypatch.setattr(sync.transaction, 'atomic', atomic)
    cursor = Mock()
    cursor.__enter__ = Mock(return_value=cursor)
    cursor.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(sync.connection, 'cursor', lambda: cursor)
    def traverse(*args):
        assert not state['atomic']
        yield row()
        assert not state['atomic']
        state['traversed'] = True
    def encode(texts):
        assert state['traversed'] and not state['atomic']
        state['encoded'] = True
        return [[1.] * sync.settings.EMBEDDING_DIMENSIONS]
    monkeypatch.setattr(sync, 'encode_texts', encode)
    def insert(objects):
        assert state['atomic']
        for obj in objects:
            obj.pk = 1
    movies.bulk_create.side_effect = insert
    monkeypatch.setattr(sync, 'refresh_search_vectors', Mock(return_value=1))
    assert sync.sync_catalog(Mock(iter_movies=traverse), today=TODAY).added == 1
    sync.EmbeddingIndex.record.assert_called_once_with(current)


@pytest.mark.django_db(transaction=True)
def test_empty_catalog_refreshes_existing_origin_record(publication):
    sync = publication
    current = sync.current_provenance()
    sync.EmbeddingIndex.record(dict(current, provider='old-provider', api_provider='old-api'))
    assert sync.sync_catalog(Mock(iter_movies=Mock(return_value=iter([row()]))), today=TODAY).added == 1
    assert sync.EmbeddingIndex.stored() == current


@pytest.mark.django_db(transaction=True)
def test_adding_new_movie_preserves_existing_search_vector(publication):
    sync = publication
    client = Mock(iter_movies=Mock(return_value=iter([row()])))
    assert sync.sync_catalog(client, today=TODAY).added == 1
    old = sync.Movie.objects.get(tmdb_id=1)
    old_vector = old.search_vector
    changed = row(serial_name='Upstream changed')
    new = row(tmdb_id=2, serial_name='Second movie')
    result = sync.sync_catalog(Mock(iter_movies=Mock(return_value=iter([changed, new]))), today=TODAY)
    assert result.added == 1
    old.refresh_from_db()
    assert old.search_vector == old_vector
    assert old.serial_name == 'Example'
    assert sync.Movie.objects.get(tmdb_id=2).search_vector is not None


@pytest.mark.parametrize('threshold', [5, 35])
def test_vote_threshold_uses_catalog_setting(settings, threshold):
    from movies.catalog_sync import _eligible
    settings.MIN_VOTE_COUNT = threshold
    assert not _eligible(row(vote_count=threshold - 1), TODAY)
    assert _eligible(row(vote_count=threshold), TODAY)
