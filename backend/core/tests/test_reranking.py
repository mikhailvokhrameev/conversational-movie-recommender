"""Reranker tests.

The cross-encoder itself is mocked throughout: these assert the blending,
truncation and fail-safe logic around the model, not the model's judgement.
No weights are downloaded, so the suite runs without network access.
"""

from unittest.mock import MagicMock, patch

import pytest

from core import reranking
from core.reranking import _movie_text, rerank_candidates


def _candidate(movie_id, total, name="Фильм", description="", genres=None):
    return {
        "id": movie_id,
        "serial_name": name,
        "genres": genres or [],
        "description": description,
        "total": total,
    }


def _mock_model(scores):
    model = MagicMock()
    model.predict.return_value = list(scores)
    return model


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Each test starts with no cached model and no sticky load failure."""
    reranking._model = None
    reranking._load_failed = False
    yield
    reranking._model = None
    reranking._load_failed = False


class TestDisabledOrNoop:
    def test_disabled_returns_input_unchanged(self, settings):
        settings.RERANKER_ENABLED = False
        candidates = [_candidate(1, 0.9), _candidate(2, 0.8)]
        assert rerank_candidates("запрос", candidates) is candidates

    def test_empty_query_returns_input_unchanged(self, settings):
        settings.RERANKER_ENABLED = True
        candidates = [_candidate(1, 0.9), _candidate(2, 0.8)]
        assert rerank_candidates("", candidates) is candidates

    def test_empty_candidates(self, settings):
        settings.RERANKER_ENABLED = True
        assert rerank_candidates("запрос", []) == []

    def test_single_candidate_skips_model(self, settings):
        """Nothing to reorder, so do not pay for a model load."""
        settings.RERANKER_ENABLED = True
        with patch.object(reranking, "get_model") as get_model:
            result = rerank_candidates("запрос", [_candidate(1, 0.9)])
        get_model.assert_not_called()
        assert len(result) == 1


class TestReranking:
    @pytest.fixture(autouse=True)
    def _enable(self, settings):
        settings.RERANKER_ENABLED = True
        settings.RERANK_TOP_K = 20
        settings.RERANK_WEIGHT = 0.5

    def test_reorders_by_cross_encoder_score(self, settings):
        """Scorer put id=1 first; the cross-encoder disagrees and outweighs it.

        Weight is 0.7 rather than the 0.5 default on purpose: with two
        candidates and exactly opposed rankings, an even blend is a tie by
        construction and would not test anything.
        """
        settings.RERANK_WEIGHT = 0.7
        candidates = [_candidate(1, 0.9), _candidate(2, 0.8)]
        with patch.object(reranking, "get_model", return_value=_mock_model([0.0, 5.0])):
            result = rerank_candidates("запрос", candidates)
        assert [c["id"] for c in result] == [2, 1]

    def test_even_blend_of_opposed_rankings_is_a_tie(self, settings):
        """Documents the default weight's behaviour rather than hiding it:
        at 0.5 neither signal can override the other outright."""
        settings.RERANK_WEIGHT = 0.5
        candidates = [_candidate(1, 0.9), _candidate(2, 0.8)]
        with patch.object(reranking, "get_model", return_value=_mock_model([0.0, 5.0])):
            result = rerank_candidates("запрос", candidates)
        assert result[0]["total"] == result[1]["total"]

    def test_truncates_to_top_k(self, settings):
        settings.RERANK_TOP_K = 2
        candidates = [_candidate(i, 1.0 - i / 10) for i in range(5)]
        with patch.object(reranking, "get_model", return_value=_mock_model([1.0, 2.0])):
            result = rerank_candidates("запрос", candidates)
        assert len(result) == 2

    def test_reranks_the_highest_scoring_slice(self, settings):
        """top_k is taken after sorting, not from input order."""
        settings.RERANK_TOP_K = 2
        candidates = [_candidate(1, 0.1), _candidate(2, 0.9), _candidate(3, 0.8)]
        model = _mock_model([1.0, 2.0])
        with patch.object(reranking, "get_model", return_value=model):
            result = rerank_candidates("запрос", candidates)
        assert {c["id"] for c in result} == {2, 3}

    def test_weight_zero_preserves_scorer_order(self, settings):
        """With no cross-encoder weight the scorer's ranking must survive."""
        settings.RERANK_WEIGHT = 0.0
        candidates = [_candidate(1, 0.9), _candidate(2, 0.8)]
        with patch.object(reranking, "get_model", return_value=_mock_model([0.0, 5.0])):
            result = rerank_candidates("запрос", candidates)
        assert [c["id"] for c in result] == [1, 2]

    def test_weight_one_ignores_scorer_order(self, settings):
        settings.RERANK_WEIGHT = 1.0
        candidates = [_candidate(1, 0.9), _candidate(2, 0.1)]
        with patch.object(reranking, "get_model", return_value=_mock_model([0.0, 5.0])):
            result = rerank_candidates("запрос", candidates)
        assert [c["id"] for c in result] == [2, 1]

    def test_records_provenance_fields(self, settings):
        candidates = [_candidate(1, 0.9), _candidate(2, 0.8)]
        with patch.object(reranking, "get_model", return_value=_mock_model([3.0, 1.0])):
            result = rerank_candidates("запрос", candidates)
        top = result[0]
        assert top["rerank_score"] == 3.0
        assert top["pre_rerank_total"] == 0.9
        assert top["total"] != 0.9

    def test_does_not_mutate_input(self, settings):
        candidates = [_candidate(1, 0.9), _candidate(2, 0.8)]
        with patch.object(reranking, "get_model", return_value=_mock_model([0.0, 5.0])):
            rerank_candidates("запрос", candidates)
        assert candidates[0]["total"] == 0.9
        assert "rerank_score" not in candidates[0]

    def test_query_is_paired_with_each_movie(self, settings):
        candidates = [_candidate(1, 0.9, name="Солярис"), _candidate(2, 0.8, name="Сталкер")]
        model = _mock_model([1.0, 2.0])
        with patch.object(reranking, "get_model", return_value=model):
            rerank_candidates("научная фантастика", candidates)
        pairs = model.predict.call_args[0][0]
        assert [p[0] for p in pairs] == ["научная фантастика"] * 2
        assert "Солярис" in pairs[0][1]


class TestFailSafe:
    @pytest.fixture(autouse=True)
    def _enable(self, settings):
        settings.RERANKER_ENABLED = True

    def test_unavailable_model_returns_candidates(self, settings):
        candidates = [_candidate(1, 0.9), _candidate(2, 0.8)]
        with patch.object(reranking, "get_model", return_value=None):
            result = rerank_candidates("запрос", candidates)
        assert [c["id"] for c in result] == [1, 2]

    def test_unavailable_model_does_not_truncate_pool_larger_than_top_k(self, settings):
        """Fail-safe intentionally returns the full pool, not the top_k slice --
        per the module docstring, a reranker failure must not cost the caller
        candidates it would otherwise have had for downstream diversification."""
        settings.RERANK_TOP_K = 2
        candidates = [_candidate(i, 1.0 - i / 10) for i in range(5)]
        with patch.object(reranking, "get_model", return_value=None):
            result = rerank_candidates("запрос", candidates)
        assert len(result) == 5

    def test_predict_failure_falls_back_to_scorer_order(self, settings):
        model = MagicMock()
        model.predict.side_effect = RuntimeError("inference exploded")
        candidates = [_candidate(1, 0.8), _candidate(2, 0.9)]
        with patch.object(reranking, "get_model", return_value=model):
            result = rerank_candidates("запрос", candidates)
        assert [c["id"] for c in result] == [2, 1]

    def test_load_failure_is_not_retried(self, settings):
        """A failed download must not be re-attempted on every request."""
        with patch.object(reranking, "settings") as mock_settings:
            mock_settings.RERANKER_MODEL = "does-not-exist"
            mock_settings.RERANK_MAX_LENGTH = 256
            with patch(
                "sentence_transformers.CrossEncoder", side_effect=OSError("no such model")
            ) as ctor:
                assert reranking.get_model() is None
                assert reranking.get_model() is None
        assert ctor.call_count == 1


class TestMovieText:
    def test_joins_populated_fields(self):
        text = _movie_text(
            {"serial_name": "Солярис", "genres": ["Фантастика"], "description": "Океан."}
        )
        assert text == "Солярис. Фантастика. Океан."

    def test_skips_empty_fields(self):
        assert _movie_text({"serial_name": "Солярис"}) == "Солярис"

    def test_truncates_long_description(self):
        text = _movie_text({"serial_name": "Ф", "description": "я" * 1000})
        assert len(text) < 500
