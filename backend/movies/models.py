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
    serial_name = models.CharField(max_length=500, db_index=True)
    genres = models.JSONField(default=list)
    content_type = models.CharField(max_length=50)
    country = models.JSONField(default=list)
    actors = models.JSONField(default=list)
    director = models.CharField(max_length=255, blank=True, default="")
    age_rating = models.FloatField(null=True, blank=True)
    studio_name = models.CharField(max_length=500, blank=True, default="")
    release_date = models.DateField(null=True, blank=True)
    description = models.TextField(blank=True, default="")
    url = models.URLField(max_length=500, unique=True)
    embedding = VectorField(
        dimensions=settings.EMBEDDING_DIMENSIONS, null=True, blank=True
    )

    # Lexical retrieval channel: title/director/actors as a weighted tsvector.
    # Populated by movies.search_index.refresh_search_vectors(), not on save --
    # the catalog is bulk-imported, so it is rebuilt in one UPDATE afterwards.
    search_vector = SearchVectorField(null=True, blank=True)

    # No ANN index on `embedding` by design. At ~18K rows an exact cosine scan
    # costs on the order of tens of milliseconds, against an ~2s Ollama intent
    # call in the same request -- so HNSW bought under 2% of request latency
    # while giving up exact recall. It also degrades under the hard filters in
    # candidate_generation: a filtered HNSW scan post-filters its candidate
    # list and can return far fewer rows than the requested limit. Revisit if
    # the catalog grows by an order of magnitude.
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
