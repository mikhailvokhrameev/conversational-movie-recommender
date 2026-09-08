import secrets
import uuid

from django.conf import settings
from django.contrib.postgres.indexes import GinIndex
from django.contrib.postgres.search import SearchVectorField
from django.db import models
from django.utils import timezone
from pgvector.django import VectorField


def generate_token():
    return secrets.token_urlsafe(32)


class Movie(models.Model):
    tmdb_id = models.IntegerField(unique=True)
    serial_name = models.CharField(max_length=500, db_index=True)
    original_title = models.CharField(max_length=500, blank=True, default="")
    genres = models.JSONField(default=list)
    country = models.JSONField(default=list)
    original_language = models.CharField(max_length=10, blank=True, default="")
    keywords = models.JSONField(default=list)
    release_date = models.DateField(null=True, blank=True)
    description = models.TextField(blank=True, default="")
    runtime = models.IntegerField(null=True, blank=True)
    popularity = models.FloatField(default=0.0)
    vote_average = models.FloatField(default=0.0)
    vote_count = models.IntegerField(default=0)
    poster_path = models.CharField(max_length=500, null=True, blank=True)
    embedding = VectorField(
        dimensions=settings.EMBEDDING_DIMENSIONS, null=True, blank=True
    )

    # Lexical retrieval channel: title/original_title/keywords as a weighted
    # tsvector. Populated by movies.search_index.refresh_search_vectors(), not
    # on save -- the catalog is bulk-imported, so it is rebuilt in one UPDATE
    # afterwards.
    search_vector = SearchVectorField(null=True, blank=True)

    # No ANN index on `embedding` by design. At ~50K rows an exact cosine scan
    # still costs on the order of tens of milliseconds, against an ~2s Ollama
    # intent call in the same request -- so HNSW buys under 2% of request
    # latency while giving up exact recall. It also degrades under the hard
    # filters in candidate_generation: a filtered HNSW scan post-filters its
    # candidate list and can return far fewer rows than the requested limit.
    # Revisit if the catalog grows by another order of magnitude.
    #
    # The GIN index below is a different case: full-text matching without an
    # index means computing tsvectors for every row on every query, which is
    # real text processing rather than a cheap dot product.
    class Meta:
        indexes = [
            GinIndex(fields=["search_vector"], name="movie_search_vector_gin"),
        ]

    def __str__(self):
        return self.serial_name


class ChatSession(models.Model):
    session_id = models.UUIDField(default=uuid.uuid4, unique=True, db_index=True)
    session_token = models.CharField(max_length=64, default=generate_token, db_index=True)
    preference_vector = VectorField(
        dimensions=settings.EMBEDDING_DIMENSIONS, null=True, blank=True
    )
    preferences = models.JSONField(default=dict)
    history = models.JSONField(default=list)
    turn_count = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["updated_at"]),
        ]

    def is_expired(self):
        return timezone.now() - self.created_at > timezone.timedelta(
            hours=settings.SESSION_TTL_HOURS
        )

    def __str__(self):
        return f"Session {self.session_id} (turn {self.turn_count})"
