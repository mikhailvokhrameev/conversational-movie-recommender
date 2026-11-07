"""Add eligible TMDB discoveries atomically, without changing existing movies.

Discovery is repeated from 2024 on every run; it is not a complete TMDB catalog
snapshot. Network traversal and inference finish before any publication lock.
"""
import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
from django.conf import settings
from django.db import connection, transaction

from core.embedding_provenance import compare_provenance, current_provenance, plan_generate
from core.embedding_service import build_movie_embedding_text, encode_texts
from movies.models import EmbeddingIndex, Movie
from movies.search_index import refresh_search_vectors
from movies.tmdb_client import TMDBClient, TMDBSyncError

logger = logging.getLogger(__name__)
_START = date(2024, 1, 1)


@dataclass(frozen=True)
class SyncResult:
    checked: int
    added: int


def _eligible(row, today):
    return (row.get('adult') is False and row.get('status') == 'Released'
            and isinstance(row.get('release_date'), date)
            and _START <= row['release_date'] <= today
            and bool(row.get('description', '').strip())
            and bool((row.get('poster_path') or '').strip())
            and row.get('vote_count', 0) >= settings.MIN_VOTE_COUNT)


def _check_index(current):
    stored = EmbeddingIndex.stored()
    # CSV imports without vectors/provenance must remain an explicit user
    # embedding operation. Never publish an isolated new embedded subset.
    if Movie.objects.exists() and (stored is None or Movie.objects.filter(embedding__isnull=True).exists()):
        logger.warning('TMDB sync skipped: existing catalog has no ready embedding index; run the explicit embedding workflow first.')
        return False
    plan_generate(compare_provenance(stored, current),
                  Movie.objects.filter(embedding__isnull=False).exists(),
                  rebuild_all=False, adopt=False)
    return True


def sync_catalog(client: TMDBClient, *, today: date) -> SyncResult:
    """Publish only new movies and their validated vectors in one transaction.

    Callers supply today's UTC date. Exceptions are sanitized so provider errors
    cannot leak credentials; detailed external responses are never logged.
    """
    try:
        if type(today) is not date or today < _START:
            raise TMDBSyncError('Invalid catalog synchronization date')
        # Exhaust before even consulting the DB: a late pagination error must
        # never publish the earlier successfully fetched pages.
        rows = list(client.iter_movies(_START, today))
        unique = {row['tmdb_id']: row for row in rows}
        checked = len(unique)
        existing = set(Movie.objects.filter(tmdb_id__in=unique).values_list('tmdb_id', flat=True))
        candidates = [row for key, row in unique.items() if key not in existing and _eligible(row, today)]
        if not candidates:
            return SyncResult(checked, 0)
        current = current_provenance()
        if not _check_index(current):
            return SyncResult(checked, 0)
        texts = [build_movie_embedding_text(row['serial_name'], row['genres'],
                                           row['description'], row['keywords']) for row in candidates]
        prepared = []
        for start in range(0, len(texts), settings.EMBEDDING_BATCH_SIZE):
            batch = texts[start:start + settings.EMBEDDING_BATCH_SIZE]
            vectors = np.asarray(encode_texts(batch), dtype=float)
            if vectors.shape != (len(batch), settings.EMBEDDING_DIMENSIONS) or not np.isfinite(vectors).all():
                raise TMDBSyncError('Invalid catalog embedding vectors')
            prepared.extend(vectors.tolist())
        with transaction.atomic():
            # Table locks cover even an empty catalog / absent singleton row,
            # unlike select_for_update on a row which may not exist yet. They
            # also serialize inserts from other catalog writers until commit.
            with connection.cursor() as cursor:
                cursor.execute('LOCK TABLE movies_embeddingindex, movies_movie IN SHARE ROW EXCLUSIVE MODE')
            if not _check_index(current):
                return SyncResult(checked, 0)
            had_embeddings = Movie.objects.filter(embedding__isnull=False).exists()
            stored = EmbeddingIndex.stored()
            existing = set(Movie.objects.filter(tmdb_id__in=unique).values_list('tmdb_id', flat=True))
            objects = [Movie(**{key: value for key, value in row.items() if key not in {'adult', 'status'}}, embedding=vector)
                       for row, vector in zip(candidates, prepared) if row['tmdb_id'] not in existing]
            if not objects:
                return SyncResult(checked, 0)
            Movie.objects.bulk_create(objects)
            ids = [obj.pk for obj in objects]
            if refresh_search_vectors(movie_ids=ids) != len(ids):
                raise TMDBSyncError('Catalog search index update was incomplete')
            # Preserve provenance of existing vectors, including unknown legacy
            # revisions. With no vectors, an allowed origin change starts a new
            # index and must record the provider which actually generated it.
            if stored is None or (not had_embeddings and compare_provenance(stored, current).status == "backend_changed"):
                EmbeddingIndex.record(current)
        return SyncResult(checked, len(objects))
    except TMDBSyncError:
        raise
    except Exception:
        raise TMDBSyncError('Catalog synchronization failed; no new movies were published') from None
