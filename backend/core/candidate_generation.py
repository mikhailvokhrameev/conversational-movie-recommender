"""Candidate generation: hard filters, two retrieval channels, RRF fusion.

Applies hard filters (genre negation, country exclusion, max age rating,
min release year, content type) as SQL WHERE clauses, then retrieves through
two independent channels and fuses their rankings:

  semantic  exact pgvector cosine distance over the query embedding
  lexical   Postgres full-text match over title/director/actors

The two exist for different failure modes. Embeddings encode meaning but not
identity, so a query naming a specific film or actor can miss it entirely.
Full-text matches names literally but understands nothing. Neither subsumes
the other.

Their scores are not comparable -- a cosine distance and a ts_rank live on
unrelated scales -- so they are fused by *rank* via Reciprocal Rank Fusion
(Cormack et al. 2009) rather than by score:

    rrf(d) = sum over channels of  weight_c / (k + rank_c(d))

RRF is scale-free by construction, which is exactly why it suits this step
and why the per-signal scoring in scoring.py uses min-max normalization
instead: there, all signals rank the same pool and magnitude is meaningful.

Fusion decides pool *membership*; scoring.py then re-ranks the pool.

Pipeline position:
  query ──> hard filters ──┬──> semantic top-N ──┐
                           └──> lexical top-N  ──┴──> RRF ──> candidates
"""

import logging
import operator
import re
from functools import reduce

from django.contrib.postgres.search import SearchQuery, SearchRank
from django.db.models import F, Q
from pgvector.django import CosineDistance

from movies.models import Movie

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATE_COUNT = 100

# RRF's rank-smoothing constant. 60 is the value from the original paper and
# the de facto default; it keeps any single channel's top hit from dominating
# the fused ordering outright.
RRF_K = 60

# Semantic outranks lexical because it is meaningful for every query, while
# lexical only carries signal when the user names something. Lexical is not
# scored below its own merit here -- it just loses ties.
RRF_WEIGHTS = {"semantic": 1.0, "lexical": 0.7}

# Tokens shorter than this are dropped from the lexical query: they are almost
# all prepositions and particles, and they broaden the OR-match for nothing.
_MIN_TERM_LENGTH = 3

# Cap on lexical query terms, so a long rambling message cannot build a
# pathologically large tsquery.
_MAX_TERMS = 12

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def generate_candidates(
    query_embedding: list[float],
    intent: dict | None = None,
    limit: int = DEFAULT_CANDIDATE_COUNT,
    query_text: str = "",
) -> list[dict]:
    """Retrieve candidates via hard-filtered semantic + lexical search.

    1. Hard filters: exclude movies matching negated genres, excluded
       countries, above the requested age rating, or older than the
       requested release year; optionally pin content_type
    2. Rank the survivors by exact cosine distance (semantic channel)
    3. Rank the survivors by full-text match (lexical channel), if
       `query_text` yields any usable terms
    4. Fuse the two rankings with RRF and take `limit`

    The filters are true SQL WHERE-clause exclusions, not scoring weights --
    a movie violating a hard constraint never reaches the candidate set,
    regardless of how well it scores on either channel.

    With no `query_text` (or no usable terms in it) this degrades to
    semantic-only retrieval, which is the behaviour that predates the
    lexical channel.
    """
    base_queryset = _apply_hard_filters(
        Movie.objects.filter(embedding__isnull=False), intent
    )

    semantic_ids = list(
        base_queryset.annotate(distance=CosineDistance("embedding", query_embedding))
        .order_by("distance")
        .values_list("id", flat=True)[:limit]
    )
    lexical_ids = _lexical_ranked_ids(base_queryset, query_text, limit)

    if lexical_ids:
        fused = _rrf_fuse({"semantic": semantic_ids, "lexical": lexical_ids})
        ordered_ids = [movie_id for movie_id, _ in fused][:limit]
    else:
        ordered_ids = semantic_ids

    if not ordered_ids:
        return []

    lexical_rank_by_id = {movie_id: rank for rank, movie_id in enumerate(lexical_ids)}

    movies_by_id = {
        movie.id: movie
        for movie in Movie.objects.filter(id__in=ordered_ids).annotate(
            distance=CosineDistance("embedding", query_embedding)
        )
    }

    return [
        _as_candidate_dict(movies_by_id[movie_id], lexical_rank_by_id.get(movie_id))
        for movie_id in ordered_ids
        if movie_id in movies_by_id
    ]


def _apply_hard_filters(queryset, intent: dict | None):
    """Apply the intent's hard constraints as SQL WHERE clauses.

    Movies with a null age_rating or release_date are kept rather than
    excluded -- unknown metadata means "cannot judge", not "fails".
    """
    if not intent:
        return queryset

    for negated_genre in intent.get("negations", []):
        queryset = queryset.exclude(genres__contains=[negated_genre])

    for excluded_country in intent.get("country_exclusions", []):
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

    return queryset


def _lexical_terms(query_text: str) -> list[str]:
    """Extract usable full-text terms from a raw user message."""
    if not query_text:
        return []
    terms = [
        term
        for term in _WORD_RE.findall(query_text.lower())
        if len(term) >= _MIN_TERM_LENGTH
    ]
    return terms[:_MAX_TERMS]


def _lexical_ranked_ids(base_queryset, query_text: str, limit: int) -> list[int]:
    """Movie ids matching the query's terms, best full-text rank first.

    Terms are OR-ed rather than AND-ed. The input is a conversational
    sentence, so requiring every term to match would find nothing; OR lets a
    title or a name carry the match on its own, and ts_rank sorts by how much
    of the query landed and in which weight class (title outranks cast).
    """
    terms = _lexical_terms(query_text)
    if not terms:
        return []

    search_query = reduce(
        operator.or_, (SearchQuery(term, config="russian") for term in terms)
    )

    return list(
        base_queryset.filter(search_vector=search_query)
        .annotate(lexical_rank=SearchRank(F("search_vector"), search_query))
        .order_by("-lexical_rank")
        .values_list("id", flat=True)[:limit]
    )


def _rrf_fuse(
    channels: dict[str, list[int]],
    k: int = RRF_K,
    weights: dict[str, float] | None = None,
) -> list[tuple[int, float]]:
    """Fuse per-channel rankings into one ordering by Reciprocal Rank Fusion.

    `channels` maps a channel name to its ranked movie ids, best first.
    Returns (movie_id, fused_score) pairs, highest score first.
    """
    channel_weights = weights or RRF_WEIGHTS
    scores: dict[int, float] = {}

    for channel, ranked_ids in channels.items():
        weight = channel_weights.get(channel, 1.0)
        for rank, movie_id in enumerate(ranked_ids):
            scores[movie_id] = scores.get(movie_id, 0.0) + weight / (k + rank + 1)

    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def _as_candidate_dict(movie: Movie, lexical_rank: int | None) -> dict:
    return {
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
        # 0-based position in the lexical channel, or None if it was found
        # only semantically. Kept so the reranker can tell an exact-name hit
        # from a purely semantic one.
        "lexical_rank": lexical_rank,
    }
