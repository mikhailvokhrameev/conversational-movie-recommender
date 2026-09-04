"""Candidate generation via exact pgvector nearest neighbor search.

Applies hard filters (genre negation, country exclusion, max age rating,
min release year, content type) as SQL WHERE clauses, then orders the
surviving rows by exact cosine distance and takes the top-N.

Exact rather than approximate: see the note on Movie.embedding in models.py.
Filters run before the ordering, so they constrain the true nearest-neighbor
set rather than post-filtering an approximate one.

Pipeline position:
  query embedding ──> hard filters ──> exact cosine top-N ──> candidates
"""

import logging

from django.db.models import Q
from pgvector.django import CosineDistance

from movies.models import Movie

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATE_COUNT = 100


def generate_candidates(
    query_embedding: list[float],
    intent: dict | None = None,
    limit: int = DEFAULT_CANDIDATE_COUNT,
) -> list[dict]:
    """Retrieve candidate movies via exact cosine search with hard filters.

    1. Hard filters: exclude movies matching negated genres, excluded
       countries, above the requested age rating, or older than the
       requested release year
    2. Optional: filter by content_type if specified in intent
    3. Order the surviving rows by exact cosine distance, take `limit`

    The filters are true SQL WHERE-clause exclusions, not scoring weights --
    a movie violating a hard constraint never reaches the candidate set,
    regardless of how well it scores semantically.
    """
    queryset = Movie.objects.filter(embedding__isnull=False)

    if intent:
        negations = intent.get("negations", [])
        for neg_genre in negations:
            queryset = queryset.exclude(genres__contains=[neg_genre])

        country_exclusions = intent.get("country_exclusions", [])
        for excluded_country in country_exclusions:
            queryset = queryset.exclude(country__contains=[excluded_country])

        max_age_rating = intent.get("max_age_rating")
        if max_age_rating is not None:
            queryset = queryset.filter(
                Q(age_rating__lte=max_age_rating) | Q(age_rating__isnull=True)
            )

        min_release_year = intent.get("min_release_year")
        if min_release_year is not None:
            queryset = queryset.filter(
                Q(release_date__year__gte=min_release_year) | Q(release_date__isnull=True)
            )

        content_type = intent.get("content_type")
        if content_type:
            queryset = queryset.filter(content_type=content_type)

    candidates = (
        queryset
        .annotate(distance=CosineDistance("embedding", query_embedding))
        .order_by("distance")[:limit]
    )

    return [
        {
            "id": movie.id,
            "serial_name": movie.serial_name,
            "genres": movie.genres,
            "content_type": movie.content_type,
            "country": movie.country,
            "actors": movie.actors,
            "director": movie.director,
            "age_rating": movie.age_rating,
            "release_date": str(movie.release_date) if movie.release_date else None,
            "description": movie.description,
            "url": movie.url,
            "embedding": list(movie.embedding) if movie.embedding is not None else None,
            "distance": float(movie.distance),
        }
        for movie in candidates
    ]
