"""Release-date bounds of the loaded catalog.

The classify prompt and the no-match reply need to know where the catalog
ends: a "newest movies" request must not become the calendar year when the
data stops earlier. The bounds load on first use and are cached for the
process; an empty catalog (DB not imported yet) yields None and is retried on
the next call. Callers on the event loop must run `catalog_year_range` in a
thread (`sync_to_async`): it uses the ORM.
"""

import datetime
import threading

_lock = threading.Lock()
_bounds: tuple[int, int, datetime.date] | None = None


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


def clear_catalog_year_cache() -> None:
    """Forget the cached bounds (tests, or after a catalog re-import)."""
    global _bounds
    with _lock:
        _bounds = None
