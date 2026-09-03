import pytest

from core.candidate_generation import generate_candidates
from movies.models import Movie

EMBEDDING = [0.1] * 768


_counter = 0


def _make_movie(**overrides):
    global _counter
    _counter += 1
    defaults = {
        "serial_name": "Test Movie",
        "genres": ["Драмы"],
        "content_type": "Фильм",
        "country": ["Россия"],
        "actors": [],
        "director": "",
        "age_rating": 16.0,
        "release_date": "2020-01-01",
        "description": "",
        "embedding": EMBEDDING,
        "url": f"https://okko.tv/movie-{_counter}",
    }
    defaults.update(overrides)
    return Movie.objects.create(**defaults)


@pytest.mark.django_db
class TestHardFilters:
    def test_negation_excludes_genre(self):
        _make_movie(serial_name="Comedy", genres=["Комедии"], url="https://okko.tv/1")
        _make_movie(serial_name="Horror", genres=["Ужасы"], url="https://okko.tv/2")

        results = generate_candidates(EMBEDDING, intent={"negations": ["Ужасы"]})

        names = {m["serial_name"] for m in results}
        assert "Comedy" in names
        assert "Horror" not in names

    def test_country_exclusion_excludes_country(self):
        _make_movie(serial_name="Russian film", country=["Россия"], url="https://okko.tv/1")
        _make_movie(serial_name="US film", country=["США"], url="https://okko.tv/2")

        results = generate_candidates(EMBEDDING, intent={"country_exclusions": ["США"]})

        names = {m["serial_name"] for m in results}
        assert "Russian film" in names
        assert "US film" not in names

    def test_max_age_rating_excludes_higher_rated(self):
        _make_movie(serial_name="Kids film", age_rating=6.0, url="https://okko.tv/1")
        _make_movie(serial_name="Adult film", age_rating=18.0, url="https://okko.tv/2")

        results = generate_candidates(EMBEDDING, intent={"max_age_rating": 12})

        names = {m["serial_name"] for m in results}
        assert "Kids film" in names
        assert "Adult film" not in names

    def test_max_age_rating_keeps_unrated_movies(self):
        _make_movie(serial_name="Unrated film", age_rating=None, url="https://okko.tv/1")

        results = generate_candidates(EMBEDDING, intent={"max_age_rating": 6})

        names = {m["serial_name"] for m in results}
        assert "Unrated film" in names

    def test_min_release_year_excludes_older_films(self):
        _make_movie(serial_name="Old film", release_date="2005-01-01", url="https://okko.tv/1")
        _make_movie(serial_name="New film", release_date="2022-01-01", url="https://okko.tv/2")

        results = generate_candidates(EMBEDDING, intent={"min_release_year": 2015})

        names = {m["serial_name"] for m in results}
        assert "New film" in names
        assert "Old film" not in names

    def test_hard_filters_combine(self):
        _make_movie(
            serial_name="Matches all",
            genres=["Комедии"],
            country=["Россия"],
            age_rating=12.0,
            release_date="2021-01-01",
            url="https://okko.tv/1",
        )
        _make_movie(
            serial_name="Wrong country",
            genres=["Комедии"],
            country=["США"],
            age_rating=12.0,
            release_date="2021-01-01",
            url="https://okko.tv/2",
        )

        results = generate_candidates(
            EMBEDDING,
            intent={
                "negations": ["Ужасы"],
                "country_exclusions": ["США"],
                "max_age_rating": 16,
                "min_release_year": 2018,
            },
        )

        names = {m["serial_name"] for m in results}
        assert "Matches all" in names
        assert "Wrong country" not in names
