"""Tests for core.evaluation metrics. Pure functions plus a mocked judge -- no DB,
no Ollama."""

from unittest.mock import patch

import httpx
import pytest

from core.evaluation import (
    aggregate_metrics,
    evaluate_query,
    judge_ndcg,
    llm_relevance_score,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestNdcgFromLabels:
    """The ideal ranking comes from the golden grades, never from the returned list."""

    def test_no_golden_items_scores_zero(self):
        assert ndcg_at_k([10, 11, 12], {1: 2, 2: 1}, k=3) == 0.0

    def test_perfect_ordering_scores_one(self):
        grades = {1: 2, 2: 1, 3: 1}
        assert ndcg_at_k([1, 2, 3], grades, k=3) == pytest.approx(1.0)

    def test_wrong_order_scores_below_one(self):
        grades = {1: 2, 2: 1}
        assert ndcg_at_k([2, 1], grades, k=2) < 1.0

    def test_ideal_counts_golden_items_missing_from_the_list(self):
        # One hit at rank 1, but the label set holds three items: the ideal
        # fills all three positions, so this is far from 1.0.
        grades = {1: 1, 2: 1, 3: 1}
        assert ndcg_at_k([1, 50, 51], grades, k=3) == pytest.approx(1 / (1 + 1 / 1.585 + 1 / 2), rel=1e-3)

    def test_ideal_truncated_to_k(self):
        grades = {i: 1 for i in range(10)}
        assert ndcg_at_k([0, 1], grades, k=2) == pytest.approx(1.0)

    def test_empty_labels_score_zero(self):
        assert ndcg_at_k([1, 2], {}, k=2) == 0.0


class TestJudgeNdcg:
    def test_all_lowest_judge_scores_give_zero(self):
        assert judge_ndcg([1, 1, 1, 1, 1], k=5) == 0.0

    def test_all_top_judge_scores_give_one(self):
        assert judge_ndcg([5, 5, 5], k=3) == pytest.approx(1.0)

    def test_ideal_is_k_positions_at_max_grade(self):
        # A short list cannot score 1.0 even if every result is perfect.
        assert judge_ndcg([5], k=3) < 1.0

    def test_mixed_scores_between_zero_and_one(self):
        assert 0.0 < judge_ndcg([3, 1, 5], k=3) < 1.0


class TestPrecisionRecall:
    def test_counts_only_golden_ids(self):
        assert precision_at_k([1, 9, 2], [1, 2, 3], k=3) == pytest.approx(2 / 3)
        assert recall_at_k([1, 9, 2], [1, 2, 3], k=3) == pytest.approx(2 / 3)


class TestEvaluateQuery:
    def test_hit_is_golden_membership(self):
        metrics = evaluate_query([1, 2, 3], {2: 1}, k=3)
        assert metrics["precision_at_k"] == pytest.approx(1 / 3)
        assert metrics["recall_at_k"] == 1.0
        assert "llm_relevance" not in metrics
        assert "judge_ndcg" not in metrics

    def test_judge_failures_are_skipped_not_averaged_as_three(self):
        metrics = evaluate_query([1, 2, 3], {}, judge_scores=[5, None, 5], k=3)
        assert metrics["llm_relevance"] == 5.0
        # An incompletely judged list has no judge_ndcg rather than a guessed one.
        assert "judge_ndcg" not in metrics

    def test_fully_judged_list_reports_judge_ndcg(self):
        metrics = evaluate_query([1, 2], {}, judge_scores=[1, 1], k=2)
        assert metrics["llm_relevance"] == 1.0
        assert metrics["judge_ndcg"] == 0.0

    def test_all_judge_calls_failed_reports_no_judge_metrics(self):
        metrics = evaluate_query([1, 2], {}, judge_scores=[None, None], k=2)
        assert "llm_relevance" not in metrics
        assert "judge_ndcg" not in metrics


class TestLlmRelevanceScore:
    MOVIE = {"serial_name": "Alien", "genres": ["Horror"], "description": "x" * 1000}

    def test_returns_parsed_score(self):
        with patch("core.evaluation._chat_sync", return_value='{"score": 4}'):
            assert llm_relevance_score("scary", self.MOVIE) == 4

    def test_strips_think_block_before_parsing(self):
        with patch("core.evaluation._chat_sync", return_value='<think>hm</think>{"score": 2}'):
            assert llm_relevance_score("scary", self.MOVIE) == 2

    @pytest.mark.parametrize("failure", [
        httpx.ConnectError("down"),
        httpx.ReadTimeout("slow"),
    ])
    def test_request_failure_returns_none(self, failure):
        with patch("core.evaluation._chat_sync", side_effect=failure):
            assert llm_relevance_score("scary", self.MOVIE) is None

    @pytest.mark.parametrize("content", [
        "not json", '{"rating": 4}', '{"score": "high"}', '{"score": 9}', '{"score": 0}', "[4]",
    ])
    def test_unusable_response_returns_none(self, content):
        with patch("core.evaluation._chat_sync", return_value=content):
            assert llm_relevance_score("scary", self.MOVIE) is None

    def test_failure_is_not_averaged_as_three(self):
        with patch("core.evaluation._chat_sync", side_effect=httpx.ConnectError("down")):
            score = llm_relevance_score("scary", self.MOVIE)
        metrics = evaluate_query([1, 2], {}, judge_scores=[5, score], k=2)
        assert metrics["llm_relevance"] == 5.0

    def test_uses_params_for_description_cap_and_timeout(self, settings):
        settings.EVAL_JUDGE_DESCRIPTION_CHARS = 10
        settings.EVAL_JUDGE_TIMEOUT_SECONDS = 7.5
        with patch("core.evaluation._chat_sync", return_value='{"score": 3}') as chat:
            llm_relevance_score("scary", self.MOVIE)
        prompt = chat.call_args.args[0][0]["content"]
        assert "x" * 10 in prompt and "x" * 11 not in prompt
        assert chat.call_args.kwargs["timeout"] == 7.5
        assert chat.call_args.kwargs["json_mode"] is True


class TestAggregate:
    def test_averages_only_queries_that_have_the_key(self):
        agg = aggregate_metrics([{"a": 1.0, "j": 0.5}, {"a": 0.0}])
        assert agg == {"a": 0.5, "j": 0.5}


class TestEvaluationSettingsValidation:
    @pytest.mark.parametrize("key", ["judge_description_chars", "judge_timeout_seconds"])
    @pytest.mark.parametrize("bad", [0, -1, "30", None, True])
    def test_rejects_non_positive_or_non_number(self, key, bad):
        from recommender.settings import _validate_evaluation

        evaluation = {"judge_description_chars": 300, "judge_timeout_seconds": 30}
        evaluation[key] = bad
        with pytest.raises(ValueError, match=f"evaluation.{key}"):
            _validate_evaluation(evaluation)

    def test_description_chars_must_be_int(self):
        from recommender.settings import _validate_evaluation

        with pytest.raises(ValueError, match="evaluation.judge_description_chars"):
            _validate_evaluation({"judge_description_chars": 1.5, "judge_timeout_seconds": 30})

    def test_accepts_configured_values(self):
        from recommender.settings import _validate_evaluation

        _validate_evaluation({"judge_description_chars": 300, "judge_timeout_seconds": 30})
        _validate_evaluation({"judge_description_chars": 1, "judge_timeout_seconds": 0.5})
