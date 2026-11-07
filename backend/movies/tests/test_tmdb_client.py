from datetime import date
import socket

import httpx
import pytest

from movies.tmdb_client import TMDBClient, TMDBSyncError

CONFIG = dict(base_url='https://api.themoviedb.org/3', timeout_seconds=1,
              max_retries=1, retry_backoff_seconds=0, max_pages=2, page_size=20)
START, END = date(2025, 1, 1), date(2025, 1, 2)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError('external networking forbidden')
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setattr(socket, 'getaddrinfo', blocked)


def detail(i=1):
    return dict(id=i, adult=False, status='Released', title='English title', original_title='Original', overview='Story',
                genres=[dict(id=1, name='Drama')], production_countries=[dict(name='France', iso_3166_1='FR')],
                keywords=dict(id=i, keywords=[dict(id=2, name='friendship')]),
                original_language='fr', release_date='2025-01-01', runtime=None,
                popularity=1.5, vote_average=7.2, vote_count=12, poster_path=None)


def page(ids, number=1, pages=1, total=None):
    return dict(page=number, total_pages=pages, total_results=len(ids) if total is None else total,
                results=[dict(id=i) for i in ids])


def client(handler):
    return TMDBClient(httpx.Client(transport=httpx.MockTransport(handler)), CONFIG, api_key='secret-key')


def test_normalization_and_duplicate_ids(caplog):
    def handle(request):
        assert request.url.params['language'] == 'en-US'
        assert request.url.params['api_key'] == 'secret-key'
        return httpx.Response(200, json=page([1, 1]) if '/discover/' in request.url.path else detail())
    rows = list(client(handle).iter_movies(START, END))
    assert len(rows) == 1
    assert rows[0] == dict(tmdb_id=1, adult=False, status='Released', serial_name='English title', original_title='Original', description='Story',
                          genres=['Drama'], country=['France'], keywords=['friendship'], original_language='fr',
                          release_date=START, runtime=None, popularity=1.5, vote_average=7.2,
                          vote_count=12, poster_path=None)
    assert 'secret-key' not in caplog.text


def test_all_pages():
    def handle(request):
        if '/discover/' in request.url.path:
            n = int(request.url.params['page'])
            return httpx.Response(200, json=page([n], n, 2, 2))
        return httpx.Response(200, json=detail(int(request.url.path.rsplit('/', 1)[1])))
    assert [r['tmdb_id'] for r in client(handle).iter_movies(START, END)] == [1, 2]


def test_split_over_limit():
    def handle(request):
        if '/discover/' not in request.url.path:
            return httpx.Response(200, json=detail())
        p = request.url.params
        if p['primary_release_date.gte'] != p['primary_release_date.lte']:
            return httpx.Response(200, json=page([1], pages=3, total=41))
        return httpx.Response(200, json=page([1]))
    assert len(list(client(handle).iter_movies(START, END))) == 1


def test_single_day_over_limit():
    with pytest.raises(TMDBSyncError, match='single day'):
        list(client(lambda r: httpx.Response(200, json=page([1], pages=3, total=41))).iter_movies(START, START))


@pytest.mark.parametrize('kind', ['429', '500', '401', 'timeout', 'json', 'api', 'missing', 'wrongpage', 'changed', 'badtotal'])
def test_failures_redacted(kind, caplog):
    def handle(r):
        if kind == 'timeout':
            raise httpx.ReadTimeout('secret-key', request=r)
        if kind.isdigit():
            return httpx.Response(int(kind), text='secret-key')
        if kind == 'json':
            return httpx.Response(200, text='secret-key')
        payload = page([1], pages=2, total=2)
        if kind == 'api': payload = dict(success=False, status_message='secret-key')
        if kind == 'missing': payload['results'] = []
        if kind == 'wrongpage': payload['page'] = 2
        if kind == 'changed' and r.url.params.get('page') == '2': payload['total_pages'] = 1
        if kind == 'badtotal': payload['total_results'] = 99
        if '/movie/1' in r.url.path: payload = detail()
        return httpx.Response(200, json=payload)
    with pytest.raises(TMDBSyncError) as exc:
        list(client(handle).iter_movies(START, END))
    assert 'secret-key' not in str(exc.value)
    assert 'secret-key' not in caplog.text


@pytest.mark.parametrize('field,value', [('genres', None), ('keywords', {}), ('runtime', -1),
                                        ('vote_count', True), ('popularity', float('inf')),
                                        ('release_date', 'bad'), ('title', None), ('id', 2)])
def test_invalid_details(field, value):
    def handle(r):
        payload = page([1]) if '/discover/' in r.url.path else detail()
        if '/discover/' not in r.url.path: payload[field] = value
        return httpx.Response(200, json=payload) if field != 'popularity' else httpx.Response(200, content=__import__('json').dumps(payload))
    with pytest.raises(TMDBSyncError):
        list(client(handle).iter_movies(START, END))


def test_retry_recovers():
    attempts = []
    def handle(r):
        attempts.append(r)
        if len(attempts) == 1: return httpx.Response(429)
        return httpx.Response(200, json=page([]))
    assert list(client(handle).iter_movies(START, END)) == []
    assert len(attempts) == 2


@pytest.mark.parametrize('key,value', [('timeout_seconds', 0), ('timeout_seconds', True),
    ('retry_backoff_seconds', -1), ('max_retries', -1), ('max_pages', 501),
    ('page_size', 21), ('base_url', 'http://example.com'), ('base_url', 'https://user:secret@example.com')])
def test_invalid_configuration(key, value):
    with pytest.raises(ValueError):
        TMDBClient(httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200))), {**CONFIG, key: value}, api_key='secret-key')


def test_log_redacts_actual_httpx_info(caplog):
    caplog.set_level('INFO', logger='httpx')
    assert list(client(lambda r: httpx.Response(200, json=page([]))).iter_movies(START, END)) == []
    assert 'secret-key' not in caplog.text
    assert '[REDACTED]' in caplog.text


def test_missing_last_page_detected():
    def handle(r):
        if '/discover/' not in r.url.path: return httpx.Response(200, json=detail())
        number = int(r.url.params['page'])
        return httpx.Response(200, json=page([1] if number == 1 else [], number, 2, 2))
    with pytest.raises(TMDBSyncError, match='missing'):
        list(client(handle).iter_movies(START, END))


@pytest.mark.parametrize('ids', [[], list(range(1, 22))])
def test_invalid_first_page_rejected_before_partitioning(ids):
    def handle(r):
        if '/discover/' not in r.url.path:
            return httpx.Response(200, json=detail())
        params = r.url.params
        if params['primary_release_date.gte'] != params['primary_release_date.lte']:
            return httpx.Response(200, json=page(ids, pages=3, total=41))
        return httpx.Response(200, json=page([1]))
    with pytest.raises(TMDBSyncError, match='missing or oversized page'):
        list(client(handle).iter_movies(START, END))


@pytest.mark.parametrize('adult,status', [(False, 'Released'), (True, 'Post Production')])
def test_normalizes_filter_metadata(adult, status):
    def handle(r):
        payload = page([1]) if '/discover/' in r.url.path else {**detail(), 'adult': adult, 'status': status}
        return httpx.Response(200, json=payload)
    row = list(client(handle).iter_movies(START, END))[0]
    assert row['adult'] is adult
    assert row['status'] == status


@pytest.mark.parametrize('field,value', [('adult', 0), ('adult', None), ('adult', 'false'),
                                         ('status', None), ('status', ''), ('status', '   '), ('status', 1)])
def test_rejects_invalid_filter_metadata(field, value):
    def handle(r):
        payload = page([1]) if '/discover/' in r.url.path else {**detail(), field: value}
        return httpx.Response(200, json=payload)
    with pytest.raises(TMDBSyncError):
        list(client(handle).iter_movies(START, END))


@pytest.mark.parametrize('field', ['adult', 'status'])
def test_rejects_missing_filter_metadata(field):
    def handle(r):
        payload = page([1]) if '/discover/' in r.url.path else detail()
        if '/discover/' not in r.url.path:
            del payload[field]
        return httpx.Response(200, json=payload)
    with pytest.raises(TMDBSyncError):
        list(client(handle).iter_movies(START, END))


@pytest.mark.parametrize('runtime,expected', [(None, None), (0, None), (90, 90)])
def test_normalizes_unknown_runtime(runtime, expected):
    def handle(request):
        payload = page([1]) if '/discover/' in request.url.path else {**detail(), 'runtime': runtime}
        return httpx.Response(200, json=payload)
    row = list(client(handle).iter_movies(START, END))[0]
    assert row['runtime'] == expected
