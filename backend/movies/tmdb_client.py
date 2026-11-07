"""Date-partitioned Discover traversal (user-supplied docs/tmdb api.md).

GET /3/discover/movie uses primary_release_date.gte/lte, page (1..500),
language=en-US and primary_release_date.asc sorting. Responses must contain
page/results/total_pages/total_results. Overfull ranges split into disjoint
inclusive day intervals; an overfull single day fails. Details are fetched at
/movie/{id}?language=en-US&append_to_response=keywords.

Discover is NOT a complete catalog enumeration or atomic snapshot: undated or
unindexed movies and concurrent catalog changes can cause omissions. Successful
traversal only verifies the reported pages, never completeness of TMDB.
"""
import logging
import math
import re
import time
from datetime import date, timedelta
from urllib.parse import urlsplit

import httpx


class TMDBSyncError(Exception):
    """A sanitized sync failure; never contains request URLs or response bodies."""


class _RedactAPIKey(logging.Filter):
    def filter(self, record):
        record.msg = re.sub(r'(api_key=)[^&\s"\']+', r'\1[REDACTED]', record.getMessage())
        record.args = ()
        return True


# httpx logs full request URLs at INFO, including v3 authentication query keys.
logging.getLogger('httpx').addFilter(_RedactAPIKey())


def validate_tmdb_config(config):
    if not isinstance(config, dict):
        raise ValueError('params.yaml: tmdb must be a mapping')
    url = config.get('base_url')
    parsed = urlsplit(url) if isinstance(url, str) else None
    if not parsed or parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('params.yaml: tmdb.base_url must be an HTTPS URL without credentials/query')
    for key, minimum, maximum in [('max_retries', 0, None), ('max_pages', 1, 500), ('page_size', 20, 20)]:
        value = config.get(key)
        if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
            raise ValueError(f'params.yaml: invalid tmdb.{key}')
    for key, positive in [('timeout_seconds', True), ('retry_backoff_seconds', False)]:
        value = config.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
            raise ValueError(f'params.yaml: invalid tmdb.{key}')


class TMDBClient:
    def __init__(self, client: httpx.Client, params: dict, *, api_key: str):
        validate_tmdb_config(params)
        if not isinstance(api_key, str) or not api_key.strip():
            raise TMDBSyncError('TMDB_API_KEY is required')
        self.client, self.params, self.api_key = client, dict(params), api_key

    def _get(self, path, **query):
        for attempt in range(self.params['max_retries'] + 1):
            retry = False
            try:
                response = self.client.get(
                    self.params['base_url'].rstrip('/') + path,
                    params={**query, 'api_key': self.api_key},
                    timeout=self.params['timeout_seconds'], follow_redirects=False,
                )
                retry = response.status_code == 429 or 500 <= response.status_code < 600
                if not retry:
                    if response.status_code != 200:
                        raise TMDBSyncError('TMDB HTTP request failed')
                    try:
                        body = response.json()
                    except (ValueError, UnicodeError):
                        raise TMDBSyncError('TMDB invalid JSON') from None
                    if not isinstance(body, dict) or body.get('success') is False or 'status_code' in body:
                        raise TMDBSyncError('TMDB invalid or unsuccessful response')
                    return body
            except httpx.RequestError:
                retry = True
            if retry and attempt < self.params['max_retries']:
                time.sleep(self.params['retry_backoff_seconds'] * (attempt + 1))
        raise TMDBSyncError('TMDB temporary request failure; retries exhausted') from None

    @staticmethod
    def _integer(value, minimum=0):
        if type(value) is not int or value < minimum:
            raise TMDBSyncError('TMDB invalid integer field')
        return value

    def _page(self, start, end, number):
        body = self._get('/discover/movie', language='en-US', page=number,
                         sort_by='primary_release_date.asc', include_adult='false', include_video='false',
                         **{'primary_release_date.gte': start.isoformat(), 'primary_release_date.lte': end.isoformat()})
        pages = self._integer(body.get('total_pages'))
        total = self._integer(body.get('total_results'))
        if self._integer(body.get('page'), 1) != number or not isinstance(body.get('results'), list):
            raise TMDBSyncError('TMDB missing or incorrect page')
        if (total == 0 and pages not in (0, 1)) or (total > 0 and not (pages <= total <= pages * self.params['page_size'])):
            raise TMDBSyncError('TMDB inconsistent pagination totals')
        if len(body['results']) > self.params['page_size'] or (total and not body['results']):
            raise TMDBSyncError('TMDB missing or oversized page')
        for row in body['results']:
            if not isinstance(row, dict):
                raise TMDBSyncError('TMDB invalid discovery row')
            self._integer(row.get('id'), 1)
        return body

    def _ids(self, start, end):
        first = self._page(start, end, 1)
        pages, total = first['total_pages'], first['total_results']
        if pages > self.params['max_pages'] or total > self.params['max_pages'] * self.params['page_size']:
            if start == end:
                raise TMDBSyncError('TMDB page limit exceeded for a single day')
            middle = start + timedelta(days=(end - start).days // 2)
            yield from self._ids(start, middle)
            yield from self._ids(middle + timedelta(days=1), end)
            return
        count = 0
        for number in range(1, max(1, pages) + 1):
            body = first if number == 1 else self._page(start, end, number)
            if (body['total_pages'], body['total_results']) != (pages, total):
                raise TMDBSyncError('TMDB pagination changed during traversal')
            rows = body['results']
            count += len(rows)
            yield from (row['id'] for row in rows)
        if count != total:
            raise TMDBSyncError('TMDB incomplete page results')

    @staticmethod
    def _text(value, maximum=None):
        if not isinstance(value, str) or (maximum is not None and len(value) > maximum):
            raise TMDBSyncError('TMDB invalid text field')
        return value

    def _names(self, rows):
        if not isinstance(rows, list):
            raise TMDBSyncError('TMDB missing name list')
        names = []
        for row in rows:
            if not isinstance(row, dict):
                raise TMDBSyncError('TMDB invalid name entry')
            name = self._text(row.get('name'))
            if not name.strip():
                raise TMDBSyncError('TMDB empty name')
            names.append(name)
        return names

    def _normalize(self, body, movie_id):
        if self._integer(body.get('id'), 1) != movie_id:
            raise TMDBSyncError('TMDB details ID mismatch')
        adult = body.get('adult')
        if type(adult) is not bool:
            raise TMDBSyncError('TMDB missing or invalid adult flag')
        status = self._text(body.get('status'))
        if not status.strip():
            raise TMDBSyncError('TMDB empty status')
        keywords = body.get('keywords')
        if not isinstance(keywords, dict) or ('id' in keywords and keywords['id'] != movie_id):
            raise TMDBSyncError('TMDB missing or mismatched keywords')
        released = self._text(body.get('release_date'))
        try:
            released = date.fromisoformat(released) if released else None
        except ValueError:
            raise TMDBSyncError('TMDB invalid release date') from None
        runtime = body.get('runtime')
        if 'runtime' not in body:
            raise TMDBSyncError('TMDB missing runtime')
        if runtime is not None:
            self._integer(runtime)
        # TMDB uses zero for unknown duration, matching CSV import semantics.
        if runtime == 0:
            runtime = None
        numbers = {}
        for key in ('popularity', 'vote_average'):
            value = body.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise TMDBSyncError('TMDB invalid numeric field')
            numbers[key] = value
        poster = body.get('poster_path')
        if 'poster_path' not in body:
            raise TMDBSyncError('TMDB missing poster field')
        if poster is not None: self._text(poster, 500)
        return dict(tmdb_id=movie_id, adult=adult, status=status, serial_name=self._text(body.get('title'), 500),
                    original_title=self._text(body.get('original_title'), 500),
                    description=self._text(body.get('overview')), genres=self._names(body.get('genres')),
                    country=self._names(body.get('production_countries')), keywords=self._names(keywords.get('keywords')),
                    original_language=self._text(body.get('original_language'), 10), release_date=released,
                    runtime=runtime, vote_count=self._integer(body.get('vote_count')), poster_path=poster, **numbers)

    def iter_movies(self, start_date: date, end_date: date):
        if type(start_date) is not date or type(end_date) is not date or start_date > end_date:
            raise TMDBSyncError('TMDB invalid date interval')
        seen = set()
        for movie_id in self._ids(start_date, end_date):
            if movie_id in seen:
                continue
            seen.add(movie_id)
            body = self._get(f'/movie/{movie_id}', language='en-US', append_to_response='keywords')
            yield self._normalize(body, movie_id)
