"""params.yaml values that settings.py refuses to load when they are wrong."""

import pytest

from recommender.settings import (
    _unchecked_text,
    _validate_intent,
    _validate_logging_level,
    _validate_placeholder_max_chars,
    _validate_unchecked_attributes,
    _validate_lexical_stopwords,
    _validate_rerank_weight,
    _validate_revision,
    _validate_score_weights,
)

GOOD_WEIGHTS = {"semantic": 0.4667, "metadata": 0.3333, "popularity": 0.2}


class TestScoreWeights:
    def test_shipped_params_pass(self):
        from django.conf import settings

        _validate_score_weights(settings.SCORE_WEIGHTS)
        assert sum(settings.SCORE_WEIGHTS.values()) == pytest.approx(1.0, abs=1e-6)

    def test_accepts_a_sum_of_one_within_tolerance(self):
        _validate_score_weights(GOOD_WEIGHTS)
        _validate_score_weights({**GOOD_WEIGHTS, "semantic": 0.4667 + 5e-7})

    @pytest.mark.parametrize("scale", [0.9, 1.1, 2.0])
    def test_rejects_a_sum_other_than_one(self, scale):
        weights = {k: v * scale for k, v in GOOD_WEIGHTS.items()}
        with pytest.raises(ValueError, match="sum to 1.0"):
            _validate_score_weights(weights)

    def test_rejects_a_sum_just_outside_tolerance(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            _validate_score_weights({**GOOD_WEIGHTS, "semantic": 0.4667 + 1e-5})

    def test_rejects_a_negative_weight_even_when_the_sum_is_one(self):
        with pytest.raises(ValueError, match=r"scoring.weights.popularity"):
            _validate_score_weights({"semantic": 0.95, "metadata": 0.2, "popularity": -0.15})

    @pytest.mark.parametrize("bad", ["0.5", None, True, [0.5]])
    def test_rejects_non_numeric_weights(self, bad):
        with pytest.raises(ValueError, match="scoring.weights.b"):
            _validate_score_weights({**GOOD_WEIGHTS, "b": bad})

    def test_rejects_unknown_or_missing_signal(self):
        for weights in ({**GOOD_WEIGHTS, "session": 0.0}, {"semantic": 0.5, "metadata": 0.5}):
            with pytest.raises(ValueError, match=r"scoring\.weights.*semantic, metadata, popularity"):
                _validate_score_weights(weights)

    @pytest.mark.parametrize("bad", [None, {}, [0.5, 0.5]])
    def test_rejects_a_missing_or_malformed_mapping(self, bad):
        with pytest.raises(ValueError, match="scoring.weights"):
            _validate_score_weights(bad)


class TestRerankWeight:
    @pytest.mark.parametrize("ok", [0, 0.0, 0.5, 1, 1.0])
    def test_accepts_the_closed_unit_interval(self, ok):
        _validate_rerank_weight({"weight": ok})

    @pytest.mark.parametrize("bad", [-0.01, 1.01, 5, "0.5", None, True])
    def test_rejects_anything_else(self, bad):
        with pytest.raises(ValueError, match="reranking.weight"):
            _validate_rerank_weight({"weight": bad})

    def test_shipped_value_passes(self):
        from django.conf import settings

        _validate_rerank_weight({"weight": settings.RERANK_WEIGHT})


class TestRevision:
    @pytest.mark.parametrize("ok", [None, "5617a9f61b028005a4858fdac845db406aefb181", "main"])
    def test_accepts_null_or_a_string(self, ok):
        _validate_revision("embedding", {"revision": ok})

    @pytest.mark.parametrize("bad", ["", "  ", 123, True, ["abc"]])
    def test_rejects_anything_else(self, bad):
        with pytest.raises(ValueError, match="reranking.revision"):
            _validate_revision("reranking", {"revision": bad})

    def test_shipped_params_pin_both_models(self):
        from django.conf import settings

        for revision in (settings.EMBEDDING_REVISION, settings.RERANKER_REVISION):
            assert isinstance(revision, str) and len(revision) == 40


class TestLexicalStopwords:
    def test_shipped_params_pass(self):
        from django.conf import settings

        assert "movie" in settings.LEXICAL_STOPWORDS
        _validate_lexical_stopwords(sorted(settings.LEXICAL_STOPWORDS))

    def test_accepts_an_empty_list(self):
        _validate_lexical_stopwords([])

    @pytest.mark.parametrize("bad", [["Movie"], ["movie", ""], [5], "movie", None])
    def test_rejects_uppercase_empty_or_malformed(self, bad):
        with pytest.raises(ValueError, match=r"retrieval\.lexical\.stopwords"):
            _validate_lexical_stopwords(bad)


class TestPlaceholderMaxChars:
    def test_shipped_params_pass(self):
        from django.conf import settings

        assert settings.OLLAMA_PLACEHOLDER_MAX_CHARS >= 1

    def test_accepts_a_positive_integer(self):
        _validate_placeholder_max_chars({"explanation": {"placeholder_max_chars": 60}})

    @pytest.mark.parametrize("bad", [0, -1, 1.5, "60", None, True])
    def test_rejects_a_non_positive_or_non_integer_value(self, bad):
        with pytest.raises(ValueError, match=r"llm\.explanation\.placeholder_max_chars"):
            _validate_placeholder_max_chars({"explanation": {"placeholder_max_chars": bad}})

    @pytest.mark.parametrize("llm", [{}, {"explanation": {}}, {"explanation": None}])
    def test_rejects_a_missing_key(self, llm):
        with pytest.raises(ValueError, match=r"llm\.explanation\.placeholder_max_chars"):
            _validate_placeholder_max_chars(llm)


class TestLoggingLevel:
    def test_shipped_params_pass(self):
        from django.conf import settings

        assert settings.LOGGING["loggers"]["core"]["level"] == settings.LOG_LEVEL

    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    def test_accepts_stdlib_level_names(self, level):
        _validate_logging_level({"level": level})

    @pytest.mark.parametrize("bad", ["info", "VERBOSE", "NOTSET", "", 20, None])
    def test_rejects_anything_else(self, bad):
        with pytest.raises(ValueError, match=r"logging\.level"):
            _validate_logging_level({"level": bad})

    @pytest.mark.parametrize("section", [{}, None])
    def test_rejects_a_missing_key(self, section):
        with pytest.raises(ValueError, match=r"logging\.level"):
            _validate_logging_level(section)


class TestUncheckedAttributes:
    def test_shipped_params_pass(self):
        from django.conf import settings

        assert "silent" in settings.CATALOG_UNCHECKED_ATTRIBUTES
        assert "black-and-white" in settings.CATALOG_UNCHECKED_ATTRIBUTES

    def test_both_texts_render_from_the_configured_list(self):
        from django.conf import settings
        from core.ollama_client import CLASSIFY_AND_PARSE_SYSTEM_PROMPT
        from movies.views import UNSUPPORTED_MESSAGE

        text = settings.CATALOG_UNCHECKED_TEXT
        assert all(a in text for a in settings.CATALOG_UNCHECKED_ATTRIBUTES)
        assert text in CLASSIFY_AND_PARSE_SYSTEM_PROMPT.format(
            genres="G", unchecked=text
        )
        assert text in UNSUPPORTED_MESSAGE.format(constraints="x", unchecked=text)

    def test_text_joins_with_and(self):
        assert _unchecked_text(["cast"]) == "cast"
        assert _unchecked_text(["cast", "awards"]) == "cast and awards"
        assert _unchecked_text(["cast", "director", "awards"]) == "cast, director and awards"

    def test_accepts_a_list_of_non_empty_strings(self):
        _validate_unchecked_attributes({"unchecked_attributes": ["cast", "silent"]})

    @pytest.mark.parametrize(
        "bad", [[], "cast", None, ["cast", ""], ["cast", 3], ["cast", "cast"], [" "]]
    )
    def test_rejects_anything_else(self, bad):
        with pytest.raises(ValueError, match=r"catalog\.unchecked_attributes"):
            _validate_unchecked_attributes({"unchecked_attributes": bad})

    def test_rejects_a_missing_key(self):
        with pytest.raises(ValueError, match=r"catalog\.unchecked_attributes"):
            _validate_unchecked_attributes({})


GOOD_RESOLUTION = {
    "genres": {"match_threshold": 0.5, "match_margin": 0.05, "synonyms": {"sci-fi": "Science Fiction"}},
    "countries": {"match_threshold": 0.5, "match_margin": 0.05, "synonyms": {}},
    "languages": {"synonyms": {"korean": "ko"}},
}


def _intent(**fields):
    resolution = {name: {**GOOD_RESOLUTION[name], **fields.get(name, {})} for name in GOOD_RESOLUTION}
    return {"resolution": resolution}


class TestIntentResolution:
    def test_shipped_params_pass(self):
        from django.conf import settings

        _validate_intent(settings.PARAMS["intent"])
        assert settings.INTENT_RESOLUTION["countries"]["synonyms"]["USA"] == "United States of America"
        assert "match_threshold" not in settings.INTENT_RESOLUTION["languages"]

    def test_accepts_a_good_config(self):
        _validate_intent(_intent())
        _validate_intent(_intent(genres={"match_margin": 0, "match_threshold": 1}))

    def test_removed_genre_match_threshold_points_to_resolution(self):
        with pytest.raises(ValueError, match=r"intent\.genre_match_threshold.*intent\.resolution\.genres"):
            _validate_intent({**_intent(), "genre_match_threshold": 0.5})

    @pytest.mark.parametrize("bad", [None, {}, {"resolution": None}, {"resolution": {}}])
    def test_rejects_a_missing_or_malformed_section(self, bad):
        with pytest.raises(ValueError, match=r"intent"):
            _validate_intent(bad)

    def test_rejects_unknown_or_missing_fields(self):
        intent = _intent()
        intent["resolution"]["moods"] = {"synonyms": {}}
        with pytest.raises(ValueError, match=r"intent\.resolution.*genres, countries, languages"):
            _validate_intent(intent)
        intent = _intent()
        del intent["resolution"]["countries"]
        with pytest.raises(ValueError, match=r"intent\.resolution.*genres, countries, languages"):
            _validate_intent(intent)

    @pytest.mark.parametrize("bad", [0, -0.1, 1.5, "0.5", None, True])
    def test_rejects_a_threshold_outside_zero_one(self, bad):
        with pytest.raises(ValueError, match=r"intent\.resolution\.countries\.match_threshold"):
            _validate_intent(_intent(countries={"match_threshold": bad}))

    @pytest.mark.parametrize("bad", [-0.01, 1, 2, "0.05", None, False])
    def test_rejects_a_margin_outside_zero_one(self, bad):
        with pytest.raises(ValueError, match=r"intent\.resolution\.genres\.match_margin"):
            _validate_intent(_intent(genres={"match_margin": bad}))

    def test_embedding_fields_need_both_threshold_and_margin(self):
        intent = _intent()
        del intent["resolution"]["genres"]["match_margin"]
        with pytest.raises(ValueError, match=r"intent\.resolution\.genres"):
            _validate_intent(intent)

    def test_languages_take_no_embedding_threshold(self):
        with pytest.raises(ValueError, match=r"intent\.resolution\.languages.*synonyms"):
            _validate_intent(_intent(languages={"match_threshold": 0.5}))

    @pytest.mark.parametrize(
        "bad", [None, ["USA"], {"USA": ""}, {"": "France"}, {"no": False}, {"USA": 1}, {1: "France"}]
    )
    def test_rejects_malformed_synonyms(self, bad):
        with pytest.raises(ValueError, match=r"intent\.resolution\.countries\.synonyms"):
            _validate_intent(_intent(countries={"synonyms": bad}))

    def test_accepts_empty_synonyms(self):
        _validate_intent(_intent(languages={"synonyms": {}}))
