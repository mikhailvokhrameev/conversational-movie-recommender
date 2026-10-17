from core.session_manager import track_explicit_preferences


class TestNoPreferenceVectorApi:
    def test_no_preference_vector_api(self):
        import core.session_manager as sm
        assert not hasattr(sm, "update_preference_vector")
        assert not hasattr(sm, "vector_for_scoring")


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
