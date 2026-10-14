import numpy as np
import pytest

from django.conf import settings

from core.session_manager import (
    track_explicit_preferences, update_preference_vector, vector_for_scoring,
)


def _cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


class TestVectorForScoring:
    def test_refinement_applies_refinement_alpha_before_scoring(self):
        current = [1.0, 0.0, 0.0]
        query = [0.0, 1.0, 0.0]
        result = vector_for_scoring(current, query, "refinement")
        expected = update_preference_vector(current, query, alpha=settings.SESSION_ALPHA_REFINEMENT)
        assert result == pytest.approx(expected)

    def test_refinement_vector_is_closer_to_query_than_old_vector(self):
        current = [1.0, 0.0, 0.0]
        query = [0.0, 1.0, 0.0]
        result = vector_for_scoring(current, query, "refinement")
        assert _cosine(result, query) > _cosine(current, query)

    def test_new_search_returns_current_vector_unchanged(self):
        current = [1.0, 0.0, 0.0]
        result = vector_for_scoring(current, [0.0, 1.0, 0.0], "new_search")
        assert result == current

    def test_first_turn_without_vector_stays_none(self):
        assert vector_for_scoring(None, [0.0, 1.0], "new_search") is None
        assert vector_for_scoring(None, [0.0, 1.0], "refinement") is None


class TestUpdatePreferenceVector:
    def test_first_call_returns_normalized_vector(self):
        embedding = [3.0, 4.0]
        result = update_preference_vector(None, embedding)
        norm = np.linalg.norm(result)
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_first_call_preserves_direction(self):
        embedding = [3.0, 4.0]
        result = update_preference_vector(None, embedding)
        assert result[0] == pytest.approx(0.6, abs=1e-6)
        assert result[1] == pytest.approx(0.8, abs=1e-6)

    def test_ema_blends_with_alpha(self):
        current = [1.0, 0.0]
        query = [0.0, 1.0]
        result = update_preference_vector(current, query, alpha=0.5)
        assert abs(result[0] - result[1]) < 0.01

    def test_result_is_normalized(self):
        current = [1.0, 0.0, 0.0]
        query = [0.0, 1.0, 0.0]
        result = update_preference_vector(current, query, alpha=0.7)
        norm = np.linalg.norm(result)
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_high_alpha_favors_new_query(self):
        current = [1.0, 0.0]
        query = [0.0, 1.0]
        result = update_preference_vector(current, query, alpha=0.9)
        assert result[1] > result[0]

    def test_zero_vector_handling(self):
        result = update_preference_vector(None, [0.0, 0.0, 0.0])
        assert all(v == 0.0 for v in result)


class TestTrackExplicitPreferences:
    def test_empty_preferences_and_intent(self):
        result = track_explicit_preferences({}, {})
        assert result == {
            "liked_genres": [],
            "disliked_genres": [],
            "themes": [],
            "reference_films": [],
        }

    def test_accumulates_genres(self):
        prefs = {"liked_genres": ["Комедии"]}
        intent = {"genres": ["Драмы"]}
        result = track_explicit_preferences(prefs, intent)
        assert result["liked_genres"] == ["Комедии", "Драмы"]

    def test_no_duplicates(self):
        prefs = {"liked_genres": ["Комедии"]}
        intent = {"genres": ["Комедии"]}
        result = track_explicit_preferences(prefs, intent)
        assert result["liked_genres"] == ["Комедии"]

    def test_negations_go_to_disliked(self):
        result = track_explicit_preferences({}, {"negations": ["Ужасы"]})
        assert result["disliked_genres"] == ["Ужасы"]

    def test_reference_films_accumulated(self):
        prefs = {"reference_films": ["Интерстеллар"]}
        intent = {"reference_films": ["Начало"]}
        result = track_explicit_preferences(prefs, intent)
        assert "Интерстеллар" in result["reference_films"]
        assert "Начало" in result["reference_films"]

    def test_themes_accumulated(self):
        result = track_explicit_preferences({}, {"themes": ["космос", "путешествия"]})
        assert result["themes"] == ["космос", "путешествия"]


class TestBuildMovieFacts:
    def test_builds_facts_in_display_order_with_year_from_release_date(self):
        from core.session_manager import build_movie_facts

        facts = build_movie_facts([
            {"serial_name": "A", "release_date": "1999-03-31", "genres": ["Drama"],
             "runtime": 136, "description": "First."},
            {"serial_name": "B", "release_date": "2004-01-01", "genres": ["Comedy", "Drama"],
             "runtime": 90, "description": "Second."},
        ], description_chars=200)
        assert facts == [
            {"title": "A", "year": 1999, "genres": ["Drama"], "runtime": 136, "description": "First."},
            {"title": "B", "year": 2004, "genres": ["Comedy", "Drama"], "runtime": 90, "description": "Second."},
        ]

    def test_truncates_description_at_storage_time(self):
        from core.session_manager import build_movie_facts

        facts = build_movie_facts(
            [{"serial_name": "A", "release_date": "1999-03-31", "genres": [], "runtime": 1,
              "description": "x" * 500}],
            description_chars=200,
        )
        assert facts[0]["description"] == "x" * 200

    def test_missing_release_date_runtime_and_description_become_none_or_empty(self):
        from core.session_manager import build_movie_facts

        facts = build_movie_facts(
            [{"serial_name": "A", "release_date": None, "genres": None, "runtime": None,
              "description": None}],
            description_chars=200,
        )
        assert facts == [{"title": "A", "year": None, "genres": [], "runtime": None, "description": ""}]

    def test_accepts_date_objects_for_release_date(self):
        import datetime
        from core.session_manager import build_movie_facts

        facts = build_movie_facts(
            [{"serial_name": "A", "release_date": datetime.date(1987, 5, 1), "genres": [],
              "runtime": 1, "description": ""}],
            description_chars=200,
        )
        assert facts[0]["year"] == 1987
