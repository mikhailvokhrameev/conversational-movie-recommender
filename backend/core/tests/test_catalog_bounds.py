"""catalog_year_range: process-cached release-date bounds of the catalog.

No DB: Movie is patched, so every test runs on the host."""

import datetime
from unittest.mock import MagicMock, patch

import pytest

from core import catalog_bounds

MOVIE = "movies.models.Movie"


@pytest.fixture(autouse=True)
def _fresh_cache():
    catalog_bounds.clear_catalog_year_cache()
    yield
    catalog_bounds.clear_catalog_year_cache()


def _movie_model(min_date, max_date):
    model = MagicMock()
    model.objects.aggregate.return_value = {"min_date": min_date, "max_date": max_date}
    return model


def test_returns_min_year_max_year_and_max_release_date():
    model = _movie_model(datetime.date(1915, 2, 8), datetime.date(2023, 9, 30))
    with patch(MOVIE, model):
        assert catalog_bounds.catalog_year_range() == (1915, 2023, datetime.date(2023, 9, 30))


def test_result_is_cached_for_the_process():
    model = _movie_model(datetime.date(1915, 2, 8), datetime.date(2023, 9, 30))
    with patch(MOVIE, model):
        first = catalog_bounds.catalog_year_range()
        second = catalog_bounds.catalog_year_range()
    assert first == second
    assert model.objects.aggregate.call_count == 1


def test_empty_catalog_is_none_and_retried_on_the_next_call():
    empty = _movie_model(None, None)
    with patch(MOVIE, empty):
        assert catalog_bounds.catalog_year_range() is None
        assert catalog_bounds.catalog_year_range() is None
    assert empty.objects.aggregate.call_count == 2
    filled = _movie_model(datetime.date(2000, 1, 1), datetime.date(2001, 1, 1))
    with patch(MOVIE, filled):
        assert catalog_bounds.catalog_year_range() == (2000, 2001, datetime.date(2001, 1, 1))
