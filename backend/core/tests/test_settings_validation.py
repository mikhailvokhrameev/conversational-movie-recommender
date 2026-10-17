"""params.yaml values that settings.py refuses to load when they are wrong."""

import pytest

from recommender.settings import (
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
