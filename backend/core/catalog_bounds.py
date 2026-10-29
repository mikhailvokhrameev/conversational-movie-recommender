"""Catalog-wide statistics: release-date bounds and mean rating.

The classify prompt and the no-match reply need to know where the catalog
ends: a "newest movies" request must not become the calendar year when the
data stops earlier. The bounds load on first use and are cached for the
process; an empty catalog (DB not imported yet) yields None and is retried on
the next call. `catalog_mean_rating` (the C of the scorer's weighted rating) is
cached the same way. Callers on the event loop must run these in a thread
(`sync_to_async`): they use the ORM.
"""

import datetime
import threading

_lock = threading.Lock()
_bounds: tuple[int, int, datetime.date] | None = None
_mean_rating: float | None = None


def catalog_year_range() -> tuple[int, int, datetime.date] | None:
    """(min_year, max_year, max_release_date) over the dated movies, or None
    when the catalog has none."""
    global _bounds
    if _bounds is not None:
        return _bounds
    with _lock:
        if _bounds is not None:
            return _bounds
        from django.db.models import Max, Min

        from movies.models import Movie

        dates = Movie.objects.aggregate(min_date=Min("release_date"), max_date=Max("release_date"))
        if dates["min_date"] is None or dates["max_date"] is None:
            return None
        _bounds = (dates["min_date"].year, dates["max_date"].year, dates["max_date"])
        return _bounds


def catalog_mean_rating() -> float | None:
    """Mean `vote_average` over the movies with at least one vote, or None
    when the catalog has none. Movies with no votes carry a placeholder 0.0
    and would drag the mean down, so they are excluded."""
    global _mean_rating
    if _mean_rating is not None:
        return _mean_rating
    with _lock:
        if _mean_rating is not None:
            return _mean_rating
        from django.db.models import Avg

        from movies.models import Movie

        mean = Movie.objects.filter(vote_count__gt=0).aggregate(mean=Avg("vote_average"))["mean"]
        if mean is None:
            return None
        _mean_rating = float(mean)
        return _mean_rating


def clear_catalog_cache() -> None:
    """Forget the cached statistics (tests, or after a catalog re-import)."""
    global _bounds, _mean_rating
    with _lock:
        _bounds = None
        _mean_rating = None
