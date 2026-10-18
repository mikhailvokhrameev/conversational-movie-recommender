"""Tests for core.evaluation metrics. Pure functions plus a mocked judge -- no DB,
no Ollama."""

from unittest.mock import patch

import httpx
import pytest

from core.evaluation import (
    aggregate_metrics,
    evaluate_query,
    gate_report,
    label_export_rows,
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
        assert chat.call_args.kwargs["call_type"] == "judge"

    def test_requests_an_integer_schema_on_the_judge_scale(self):
        from core.evaluation import JUDGE_MAX_SCORE, JUDGE_MIN_SCORE, JUDGE_SCHEMA

        with patch("core.evaluation._chat_sync", return_value='{"score": 3}') as chat:
            llm_relevance_score("scary", self.MOVIE)
        schema = chat.call_args.kwargs["schema"]
        assert schema is JUDGE_SCHEMA
        assert schema["required"] == ["score"]
        assert schema["properties"]["score"] == {
            "type": "integer", "minimum": JUDGE_MIN_SCORE, "maximum": JUDGE_MAX_SCORE,
        }
        assert (JUDGE_MIN_SCORE, JUDGE_MAX_SCORE) == (1, 5)


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


def _gq(returned, pool_max, unscorable=False):
    """A per-query result as evaluate_scoring records it: `returned` is a list
    of (rerank_score, is_hit)."""
    return {
        "unscorable": unscorable,
        "recommended": [{"rerank_score": score, "hit": hit} for score, hit in returned],
        "pool_max_rerank_score": pool_max,
    }


class TestGateReport:
    """Pure function: which hits / queries a candidate min_score would remove."""

    def test_distributions_split_hits_from_non_hits(self):
        queries = [
            _gq([(0.9, True), (0.5, False), (0.1, False)], 0.9),
            _gq([(0.7, True), (0.3, False)], 0.7),
        ]
        report = gate_report(queries, thresholds=[0.5])
        assert report["hits"] == {"count": 2, "min": 0.7, "q1": 0.75, "median": 0.8, "q3": 0.85, "max": 0.9}
        assert report["non_hits"]["count"] == 3
        assert report["non_hits"]["min"] == 0.1
        assert report["non_hits"]["median"] == 0.3
        assert report["non_hits"]["max"] == 0.5

    def test_threshold_row_shares(self):
        queries = [
            _gq([(0.9, True), (0.2, True), (0.1, False)], 0.9),
            _gq([(0.3, True), (0.05, False)], 0.3),
            _gq([(0.04, False)], 0.04),
        ]
        report = gate_report(queries, thresholds=[0.25, 0.5])
        low, high = report["thresholds"]
        assert low["min_score"] == 0.25
        # hits 0.9 / 0.2 / 0.3: only 0.2 is below 0.25
        assert low["hits_dropped_share"] == pytest.approx(1 / 3)
        # non-hits 0.1 / 0.05 / 0.04 all below
        assert low["non_hits_dropped_share"] == 1.0
        # a query empties when even its best pool score is below the threshold
        assert low["queries_emptied_share"] == pytest.approx(1 / 3)
        assert high["hits_dropped_share"] == pytest.approx(2 / 3)
        assert high["queries_emptied_share"] == pytest.approx(2 / 3)

    def test_threshold_is_inclusive_like_the_gate(self):
        report = gate_report([_gq([(0.5, True)], 0.5)], thresholds=[0.5])
        row = report["thresholds"][0]
        assert row["hits_dropped_share"] == 0.0
        assert row["queries_emptied_share"] == 0.0

    def test_empty_pool_max_uses_pool_not_returned_list(self):
        # The gate runs on the whole pool before MMR, so a query is empty iff the
        # pool's best score is below the threshold, whatever MMR returned.
        report = gate_report([_gq([(0.2, False)], 0.8)], thresholds=[0.5])
        assert report["thresholds"][0]["queries_emptied_share"] == 0.0

    def test_unscorable_and_unscored_queries_are_excluded_and_counted(self):
        queries = [
            _gq([(0.9, True)], 0.9),
            _gq([(0.1, False)], 0.1, unscorable=True),
            _gq([(None, False)], None),  # reranker produced no scores
        ]
        report = gate_report(queries, thresholds=[0.5])
        assert report["queries_considered"] == 1
        assert report["queries_without_scores"] == 1
        assert report["queries_unscorable"] == 1
        assert report["non_hits"]["count"] == 0
        assert report["thresholds"][0]["queries_emptied_share"] == 0.0

    def test_unscored_movies_inside_a_scored_query_are_skipped(self):
        report = gate_report([_gq([(0.9, True), (None, False)], 0.9)], thresholds=[0.5])
        assert report["non_hits"]["count"] == 0

    def test_no_data_gives_null_shares_not_zero_division(self):
        report = gate_report([], thresholds=[0.5])
        assert report["hits"]["count"] == 0
        assert report["hits"]["min"] is None
        row = report["thresholds"][0]
        assert row["hits_dropped_share"] is None
        assert row["queries_emptied_share"] is None

    def test_default_grid_is_sorted_and_inside_the_sigmoid_range(self):
        grid = [row["min_score"] for row in gate_report([])["thresholds"]]
        assert grid == sorted(grid)
        assert all(0 < t < 1 for t in grid)


def _rec(tmdb_id, score, hit=False, title=None, release_date="1999-03-31", overview="An overview."):
    return {
        "tmdb_id": tmdb_id, "title": title or f"Film {tmdb_id}", "release_date": release_date,
        "rerank_score": score, "hit": hit, "overview": overview,
    }


class TestLabelExportRows:
    def _queries(self):
        return [
            {
                "id": "d1", "query": "something funnier", "context": ["a heist film", "ok"],
                "query_text": "a funnier heist film",
                "recommended": [_rec(5, 0.91234567891234, hit=True), _rec(7, None, release_date=None)],
            },
            {
                "id": "d2", "query": "q2", "context": [], "query_text": "q2",
                "recommended": [_rec(9, 0.1, release_date="2001-01-01", overview="Multi\nline, \"quoted\"")],
            },
        ]

    def test_columns_and_order(self):
        rows = label_export_rows(self._queries())
        assert list(rows[0]) == [
            "query_id", "query", "context", "ranked_query", "rank", "tmdb_id", "title",
            "year", "overview", "rerank_score", "golden_hit", "label",
        ]
        assert [(r["query_id"], r["rank"], r["tmdb_id"]) for r in rows] == [("d1", 1, 5), ("d1", 2, 7), ("d2", 1, 9)]

    def test_query_fields_and_context_join(self):
        row = label_export_rows(self._queries())[0]
        assert row["query"] == "something funnier"
        assert row["ranked_query"] == "a funnier heist film"
        assert row["context"] == "a heist film | ok"
        assert label_export_rows(self._queries())[2]["context"] == ""

    def test_score_is_full_precision_or_blank(self):
        rows = label_export_rows(self._queries())
        assert rows[0]["rerank_score"] == repr(0.91234567891234)
        assert rows[1]["rerank_score"] == ""

    def test_golden_hit_and_empty_label(self):
        rows = label_export_rows(self._queries())
        assert [r["golden_hit"] for r in rows] == [1, 0, 0]
        assert all(r["label"] == "" for r in rows)

    def test_year_from_release_date_blank_when_missing(self):
        rows = label_export_rows(self._queries())
        assert [r["year"] for r in rows] == ["1999", "", "2001"]

    def test_overview_kept_verbatim(self):
        assert label_export_rows(self._queries())[2]["overview"] == 'Multi\nline, "quoted"'

    def test_missing_overview_is_blank(self):
        q = self._queries()
        del q[0]["recommended"][0]["overview"]
        assert label_export_rows(q)[0]["overview"] == ""
