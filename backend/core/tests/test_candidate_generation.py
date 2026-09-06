import pytest

from django.conf import settings

from core.candidate_generation import (
    _lexical_terms,
    _rrf_fuse,
    generate_candidates,
)
from movies.models import Movie
from movies.search_index import refresh_search_vectors

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

    def test_min_release_year_keeps_movies_with_null_release_date(self):
        _make_movie(serial_name="Unknown date film", release_date=None, url="https://okko.tv/3")

        results = generate_candidates(EMBEDDING, intent={"min_release_year": 2015})

        names = {m["serial_name"] for m in results}
        assert "Unknown date film" in names

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


class TestLexicalTerms:
    def test_drops_short_tokens(self):
        assert _lexical_terms("я и он в дом") == ["дом"]

    def test_lowercases_and_splits_on_punctuation(self):
        assert _lexical_terms("Место встречи, изменить нельзя!") == [
            "место",
            "встречи",
            "изменить",
            "нельзя",
        ]

    def test_empty_query(self):
        assert _lexical_terms("") == []
        assert _lexical_terms("   ") == []

    def test_caps_term_count(self):
        assert len(_lexical_terms(" ".join(f"слово{i}" for i in range(50)))) == 12


class TestRRFFuse:
    def test_single_channel_preserves_order(self):
        fused = _rrf_fuse({"semantic": [10, 20, 30]}, weights={"semantic": 1.0})
        assert [movie_id for movie_id, _ in fused] == [10, 20, 30]

    def test_document_in_both_channels_outranks_single_channel_leader(self):
        """The whole point of fusion: agreement across channels beats being
        top of one list."""
        fused = _rrf_fuse(
            {"semantic": [1, 2], "lexical": [2, 3]},
            weights={"semantic": 1.0, "lexical": 1.0},
        )
        assert fused[0][0] == 2

    def test_weights_break_ties_toward_heavier_channel(self):
        """Both are rank 0 in their own channel; semantic's weight wins."""
        fused = _rrf_fuse(
            {"semantic": [1], "lexical": [2]},
            weights={"semantic": 1.0, "lexical": 0.7},
        )
        assert [movie_id for movie_id, _ in fused] == [1, 2]

    def test_score_matches_rrf_formula(self):
        fused = dict(_rrf_fuse({"semantic": [7]}, weights={"semantic": 1.0}))
        assert fused[7] == pytest.approx(1.0 / (settings.RRF_K + 1))

    def test_empty_channels(self):
        assert _rrf_fuse({"semantic": [], "lexical": []}) == []


@pytest.mark.django_db
class TestLexicalRetrieval:
    """Exercises the real Postgres full-text path, so it needs the DB."""

    def test_exact_title_match_enters_candidate_set(self):
        """The bug this channel exists to fix: a title the embedding misses.

        "Something Unrelated" is the nearer neighbour by embedding; the named
        title is not. The user named it, so it must still be retrieved.
        """
        _make_movie(serial_name="Место встречи изменить нельзя", embedding=[0.9] * 768)
        _make_movie(serial_name="Something Unrelated", embedding=EMBEDDING)
        refresh_search_vectors()

        results = generate_candidates(
            EMBEDDING, intent=None, query_text="хочу что-то как Место встречи изменить нельзя"
        )

        names = {m["serial_name"] for m in results}
        assert "Место встречи изменить нельзя" in names

    def test_actor_name_is_matched(self):
        _make_movie(serial_name="Фильм А", actors=["Константин Хабенский"])
        _make_movie(serial_name="Фильм Б", actors=["Кто-то Другой"])
        refresh_search_vectors()

        results = generate_candidates(EMBEDDING, intent=None, query_text="фильмы с Хабенским")

        lexical_hits = [m["serial_name"] for m in results if m["lexical_rank"] is not None]
        assert "Фильм А" in lexical_hits

    def test_hard_filters_still_apply_to_lexical_hits(self):
        """A lexical match must not smuggle a movie past an exclusion."""
        _make_movie(serial_name="Место встречи изменить нельзя", country=["США"])
        refresh_search_vectors()

        results = generate_candidates(
            EMBEDDING,
            intent={"country_exclusions": ["США"]},
            query_text="Место встречи изменить нельзя",
        )

        assert results == []

    def test_no_query_text_falls_back_to_semantic_only(self):
        _make_movie(serial_name="Фильм", embedding=EMBEDDING)
        refresh_search_vectors()

        results = generate_candidates(EMBEDDING, intent=None, query_text="")

        assert len(results) == 1
        assert results[0]["lexical_rank"] is None

    def test_unmatched_query_text_falls_back_to_semantic_only(self):
        _make_movie(serial_name="Фильм", embedding=EMBEDDING)
        refresh_search_vectors()

        results = generate_candidates(
            EMBEDDING, intent=None, query_text="зззнесуществующийтокен"
        )

        assert len(results) == 1
        assert results[0]["lexical_rank"] is None
