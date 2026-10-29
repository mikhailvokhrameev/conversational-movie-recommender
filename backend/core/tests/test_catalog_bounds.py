"""catalog_year_range / catalog_mean_rating: process-cached catalog statistics.

No DB: Movie is patched, so every test runs on the host."""

import datetime
from unittest.mock import MagicMock, patch

import pytest

from core import catalog_bounds

MOVIE = "movies.models.Movie"


@pytest.fixture(autouse=True)
def _fresh_cache():
    catalog_bounds.clear_catalog_cache()
    yield
    catalog_bounds.clear_catalog_cache()


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


def _rating_model(mean):
    model = MagicMock()
    model.objects.filter.return_value.aggregate.return_value = {"mean": mean}
    return model


def test_mean_rating_is_the_mean_over_rated_movies():
    model = _rating_model(6.31)
    with patch(MOVIE, model):
        assert catalog_bounds.catalog_mean_rating() == 6.31
    model.objects.filter.assert_called_once_with(vote_count__gt=0)


def test_mean_rating_is_cached_for_the_process():
    model = _rating_model(6.31)
    with patch(MOVIE, model):
        catalog_bounds.catalog_mean_rating()
        catalog_bounds.catalog_mean_rating()
    assert model.objects.filter.call_count == 1


def test_mean_rating_none_for_empty_catalog_and_retried():
    with patch(MOVIE, _rating_model(None)):
        assert catalog_bounds.catalog_mean_rating() is None
    with patch(MOVIE, _rating_model(6.0)):
        assert catalog_bounds.catalog_mean_rating() == 6.0


def test_clear_catalog_cache_forgets_the_mean_rating():
    with patch(MOVIE, _rating_model(6.0)):
        assert catalog_bounds.catalog_mean_rating() == 6.0
    catalog_bounds.clear_catalog_cache()
    with patch(MOVIE, _rating_model(7.0)):
        assert catalog_bounds.catalog_mean_rating() == 7.0
