"""Tests for the evaluate_scoring management command.

No DB and no models: the catalog stats, golden-reference resolution, encoder,
LLM, and pipeline are patched at the command module, so these exercise the
command's own logic -- intent modes, relevance, splits,
counters, and the run record."""

import json
import subprocess
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from core.ollama_client import NON_FILTER_FIELDS, MessageIntent
from movies.management.commands import evaluate_scoring as cmd

MODULE = "movies.management.commands.evaluate_scoring"


def _movie(tmdb_id, genres=("Drama",), title=None):
    return {
        "id": tmdb_id, "tmdb_id": tmdb_id, "serial_name": title or f"Film {tmdb_id}",
        "genres": list(genres), "release_date": "2000-01-01", "description": "d",
        "embedding": [float(tmdb_id), 1.0], "total": 0.5,
    }


def _vec(text):
    """Deterministic, distinct fake embedding per text."""
    return [float(len(text)), float(sum(map(ord, text)) % 97), 1.0]


def _write_golden(tmp_path, entries):
    path = tmp_path / "golden.json"
    path.write_text(json.dumps({"_note": "draft, needs review", "queries": entries}))
    return path


def _entry(id_="q1", split="test", **extra):
    entry = {"id": id_, "query": f"query {id_}", "split": split,
             "relevant": [{"title": "Golden", "year": 2000}]}
    entry.update(extra)
    return entry


def _resolver(grades=None, unresolved=()):
    grades = grades if grades is not None else {1: 2}
    return MagicMock(return_value=(dict(grades), list(unresolved)))


class _Run:
    """Patches every external dependency of the command and runs it."""

    def __init__(self, tmp_path, entries, movies=None, intent=None, grades=None,
                 unresolved=(), judge=None):
        self.tmp_path = tmp_path
        self.golden = _write_golden(tmp_path, entries)
        self.output = tmp_path / "run.json"
        movies = movies if movies is not None else [_movie(1), _movie(2)]
        self.pipeline = MagicMock(return_value=(movies, movies))
        self.classify = AsyncMock(return_value=intent or MessageIntent(
            category="new_search", semantic_query="rewritten query",
            standalone_query="rewritten standalone query", genres=["Drama"],
        ))
        self.encode = MagicMock(side_effect=_vec)
        self.resolve = _resolver(grades, unresolved)
        self.judge = MagicMock(side_effect=judge) if judge is not None else MagicMock(return_value=3)
        self.available = AsyncMock(return_value=True)

    def __call__(self, *args):
        out = StringIO()
        with patch(f"{MODULE}._catalog_stats", return_value=({"Drama": 0.5}, 100)), \
             patch(f"{MODULE}.resolve_references", self.resolve), \
             patch(f"{MODULE}.encode_query", self.encode), \
             patch(f"{MODULE}.aclassify_and_parse", self.classify), \
             patch(f"{MODULE}.generate_and_score", self.pipeline), \
             patch(f"{MODULE}.llm_relevance_score", self.judge), \
             patch(f"{MODULE}.ais_available", self.available):
            call_command(
                "evaluate_scoring", "--golden-set", str(self.golden),
                "--output", str(self.output), *args, stdout=out,
            )
        self.stdout = out.getvalue()
        self.record = json.loads(self.output.read_text())
        return self.record


class TestE2eIntent:
    def test_calls_classify_and_parse_and_embeds_semantic_query(self, tmp_path):
        run = _Run(tmp_path, [_entry()])
        record = run()

        run.classify.assert_awaited_once_with("query q1", [])
        run.encode.assert_called_once_with("rewritten query")
        args, kwargs = run.pipeline.call_args
        assert args[0] == _vec("rewritten query")
        assert args[1] == MessageIntent(
            category="new_search", genres=["Drama"],
        ).model_dump(exclude=set(NON_FILTER_FIELDS))
        assert record["intent_mode"] == "e2e"
        assert record["queries"][0]["predicted_category"] == "new_search"

    def test_e2e_passes_standalone_query_as_query_text(self, tmp_path):
        run = _Run(tmp_path, [_entry()])
        run()
        assert run.pipeline.call_args.kwargs["query_text"] == "rewritten standalone query"

    def test_oracle_keeps_the_raw_query_as_query_text(self, tmp_path):
        run = _Run(tmp_path, [_entry()])
        record = run("--intent", "oracle")
        assert run.pipeline.call_args.kwargs["query_text"] == "query q1"
        assert record["queries"][0]["query_text"] == "query q1"
        assert record["queries"][0]["semantic_query"] is None

    def test_record_has_semantic_query_and_query_text(self, tmp_path):
        record = _Run(tmp_path, [_entry()])()
        query = record["queries"][0]
        assert query["semantic_query"] == "rewritten query"
        assert query["query_text"] == "rewritten standalone query"

    def test_context_becomes_user_history(self, tmp_path):
        run = _Run(tmp_path, [_entry(context=["first", "second"], oracle_standalone_query="x")])
        run()
        run.classify.assert_awaited_once_with(
            "query q1", [{"role": "user", "content": "first"}, {"role": "user", "content": "second"}],
        )

    @pytest.mark.parametrize("category", ["follow_up", "general_chat", "clarify"])
    def test_non_search_category_still_searches_and_counts_misclassified(self, tmp_path, category):
        run = _Run(tmp_path, [_entry()], intent=MessageIntent(category=category, semantic_query="x"))
        record = run()
        run.pipeline.assert_called_once()
        assert record["counters"]["misclassified"] == 1
        assert record["queries"][0]["misclassified"] is True


class TestUnsupportedConstraints:
    def test_still_searched_but_counted_and_flagged(self, tmp_path):
        intent = MessageIntent(
            category="new_search", semantic_query="x", unsupported_constraints=["with Keanu Reeves"],
        )
        run = _Run(tmp_path, [_entry(), _entry(id_="q2")], intent=intent)
        record = run()
        assert run.pipeline.call_count == 2
        assert "unsupported_constraints" not in run.pipeline.call_args.args[1]
        assert record["counters"]["unsupported_constraints"] == 2
        assert record["queries"][0]["unsupported_constraints"] == ["with Keanu Reeves"]
        assert "UNSUPPORTED(with Keanu Reeves)" in run.stdout

    def test_plain_parse_counts_nothing(self, tmp_path):
        record = _Run(tmp_path, [_entry()])()
        assert record["counters"]["unsupported_constraints"] == 0
        assert record["queries"][0]["unsupported_constraints"] == []

    def test_oracle_records_no_unsupported_constraints(self, tmp_path):
        record = _Run(tmp_path, [_entry()])("--intent", "oracle")
        assert record["counters"]["unsupported_constraints"] == 0
        assert "unsupported_constraints" not in record["queries"][0]


class TestParseFallbackDetection:
    """aclassify_and_parse never raises: on any failure it returns
    new_search + the raw message + no filters. e2e must not pass that off as a
    parse."""

    def test_unreachable_ollama_refuses_e2e(self, tmp_path):
        run = _Run(tmp_path, [_entry()])
        run.available.return_value = False
        with pytest.raises(CommandError, match="Ollama"):
            run()
        run.classify.assert_not_awaited()
        run.pipeline.assert_not_called()

    def test_oracle_does_not_need_ollama(self, tmp_path):
        run = _Run(tmp_path, [_entry()])
        run.available.return_value = False
        run("--intent", "oracle")
        run.available.assert_not_awaited()

    def test_fallback_shaped_intent_is_counted(self, tmp_path):
        fallback = MessageIntent(
            category="new_search", semantic_query="query q1", standalone_query="query q1",
        )
        run = _Run(tmp_path, [_entry()], intent=fallback)
        record = run()
        assert record["queries"][0]["possible_parse_fallback"] is True
        assert record["counters"]["possible_parse_fallbacks"] == 1
        assert "possible_parse_fallbacks: 1" in run.stdout

    def test_real_parse_is_not_counted(self, tmp_path):
        record = _Run(tmp_path, [_entry()])()
        assert record["queries"][0]["possible_parse_fallback"] is False
        assert record["counters"]["possible_parse_fallbacks"] == 0

    def test_raw_query_with_a_filter_is_not_counted(self, tmp_path):
        parsed = MessageIntent(category="new_search", semantic_query="query q1", genres=["Drama"])
        record = _Run(tmp_path, [_entry()], intent=parsed)()
        assert record["counters"]["possible_parse_fallbacks"] == 0

    def test_oracle_records_no_fallback_flag(self, tmp_path):
        record = _Run(tmp_path, [_entry()])("--intent", "oracle")
        assert "possible_parse_fallback" not in record["queries"][0]
        assert record["counters"]["possible_parse_fallbacks"] == 0


class TestOracleIntent:
    def test_uses_oracle_intent_and_raw_query(self, tmp_path):
        oracle = {"genres": ["Comedy"], "negations": ["Horror"]}
        run = _Run(tmp_path, [_entry(oracle_intent=oracle)])
        record = run("--intent", "oracle")

        run.classify.assert_not_awaited()
        run.encode.assert_called_once_with("query q1")
        assert run.pipeline.call_args.args[1] == oracle
        assert record["counters"]["misclassified"] == 0

    def test_context_entry_ranks_and_embeds_on_the_standalone_query(self, tmp_path):
        entry = _entry(context=["a sci-fi movie"], oracle_standalone_query="a funny science fiction movie")
        run = _Run(tmp_path, [entry])
        record = run("--intent", "oracle")

        run.encode.assert_called_once_with("a funny science fiction movie")
        assert run.pipeline.call_args.args[0] == _vec("a funny science fiction movie")
        assert run.pipeline.call_args.kwargs["query_text"] == "a funny science fiction movie"
        assert record["queries"][0]["query_text"] == "a funny science fiction movie"

    def test_missing_oracle_intent_is_empty(self, tmp_path):
        run = _Run(tmp_path, [_entry()])
        run("--intent", "oracle")
        assert run.pipeline.call_args.args[1] == {}


class TestRelevance:
    def test_genre_match_without_golden_membership_is_not_a_hit(self, tmp_path):
        oracle = {"genres": ["Comedy"]}
        movies = [_movie(50, genres=["Comedy"]), _movie(51, genres=["Comedy"])]
        run = _Run(tmp_path, [_entry(oracle_intent=oracle)], movies=movies, grades={1: 1})
        record = run("--intent", "oracle", "--k", "2")

        query = record["queries"][0]
        assert query["hits"] == []
        assert query["metrics"]["precision_at_k"] == 0.0
        assert query["metrics"]["ndcg_at_k"] == 0.0

    def test_golden_member_is_a_hit_with_its_grade(self, tmp_path):
        movies = [_movie(1), _movie(9)]
        run = _Run(tmp_path, [_entry()], movies=movies, grades={1: 2, 3: 1})
        record = run("--k", "2")
        query = record["queries"][0]
        assert query["hits"] == [1]
        assert query["metrics"]["precision_at_k"] == 0.5
        assert query["metrics"]["recall_at_k"] == 0.5

    def test_negation_violations_counted_separately(self, tmp_path):
        movies = [_movie(1, genres=["Horror"]), _movie(2, genres=["Drama"])]
        run = _Run(tmp_path, [_entry(negated_genres=["Horror"])], movies=movies, grades={1: 1})
        record = run("--k", "2")
        query = record["queries"][0]
        assert query["metrics"]["negation_violations"] == 1
        # Relevance is golden membership only; the safety metric does not change it.
        assert query["hits"] == [1]


class TestUnresolvedReferences:
    def test_reported_per_query_and_counted(self, tmp_path):
        missing = {"title": "Nowhere", "year": 1999}
        run = _Run(tmp_path, [_entry()], grades={1: 1}, unresolved=[missing])
        record = run()
        assert record["queries"][0]["unresolved_refs"] == ["Nowhere (1999)"]
        assert record["counters"]["unresolved_refs"] == 1
        assert "Nowhere (1999)" in run.stdout

    def test_query_with_nothing_resolved_is_not_scored(self, tmp_path):
        run = _Run(tmp_path, [_entry()], grades={}, unresolved=[{"title": "Golden", "year": 2000}])
        record = run()
        query = record["queries"][0]
        assert query["unscorable"] is True
        assert "precision_at_k" not in query["metrics"]
        assert record["counters"]["unscorable_queries"] == 1
        assert "precision_at_k" not in record["aggregate"]


class TestJudge:
    def test_failures_are_counted_and_not_averaged(self, tmp_path):
        run = _Run(tmp_path, [_entry()], judge=[5, None])
        record = run("--llm-judge", "--k", "2")
        assert record["counters"]["judge_failures"] == 1
        assert record["queries"][0]["judge_scores"] == [5, None]
        assert record["aggregate"]["llm_relevance"] == 5.0
        assert "judge_ndcg" not in record["aggregate"]

    def test_all_lowest_scores_give_zero_judge_ndcg(self, tmp_path):
        run = _Run(tmp_path, [_entry()], judge=[1, 1])
        record = run("--llm-judge", "--k", "2")
        assert record["aggregate"]["judge_ndcg"] == 0.0

    def test_multi_turn_judge_sees_the_conversation(self, tmp_path):
        run = _Run(tmp_path, [_entry(context=["a sci-fi movie"], oracle_standalone_query="x")], judge=[3, 3])
        run("--llm-judge", "--k", "2")
        assert "a sci-fi movie" in run.judge.call_args.args[0]
        assert "query q1" in run.judge.call_args.args[0]


class TestSplits:
    def _entries(self):
        return [_entry("d1", "dev"), _entry("d2", "dev"), _entry("t1", "test")]

    def test_plain_eval_defaults_to_test(self, tmp_path):
        record = _Run(tmp_path, self._entries())()
        assert record["split"] == "test"
        assert [q["id"] for q in record["queries"]] == ["t1"]

    def test_split_all(self, tmp_path):
        record = _Run(tmp_path, self._entries())("--split", "all")
        assert len(record["queries"]) == 3

    def test_sweep_defaults_to_dev(self, tmp_path):
        record = _Run(tmp_path, self._entries())("--sweep", "--intent", "oracle")
        assert record["split"] == "dev"
        assert record["mode"] == "sweep"
        assert record["configs"]
        assert record["best_weights"] == record["configs"][0]["weights"]
        for config in record["configs"]:
            assert sum(config["weights"].values()) == pytest.approx(1.0, abs=1e-9)
        assert {q["id"] for q in record["configs"][0]["queries"]} == {"d1", "d2"}

    @pytest.mark.parametrize("split", ["test", "all"])
    def test_sweep_refuses_anything_but_dev(self, tmp_path, split):
        with pytest.raises(CommandError, match="dev"):
            _Run(tmp_path, self._entries())("--sweep", "--split", split)

    def test_sweep_parses_each_query_once(self, tmp_path):
        run = _Run(tmp_path, self._entries())
        record = run("--sweep")
        assert run.classify.await_count == 2
        assert run.pipeline.call_count == 2 * len(record["configs"])


class TestGateReport:
    """--gate-report: dev-split calibration of reranking.min_score with the gate
    off for the run."""

    def _entries(self):
        return [_entry("d1", "dev"), _entry("d2", "dev"), _entry("t1", "test")]

    @staticmethod
    def _scored_movies():
        movies = [_movie(1), _movie(2), _movie(3)]
        for movie, score in zip(movies, (0.9, 0.2, 0.05)):
            movie["rerank_score"] = score
        return movies

    def test_defaults_to_dev_and_records_the_report(self, tmp_path):
        run = _Run(tmp_path, self._entries(), movies=self._scored_movies(), grades={1: 2})
        record = run("--gate-report", "--intent", "oracle")

        assert record["mode"] == "gate_report"
        assert record["split"] == "dev"
        assert {q["id"] for q in record["queries"]} == {"d1", "d2"}
        report = record["gate_report"]
        assert report["queries_considered"] == 2
        # tmdb 1 is the only labelled movie: one hit (0.9), two non-hits per query
        assert report["hits"]["count"] == 2 and report["hits"]["min"] == 0.9
        assert report["non_hits"]["count"] == 4 and report["non_hits"]["max"] == 0.2
        assert all("hits_dropped_share" in row for row in report["thresholds"])
        assert record["queries"][0]["pool_max_rerank_score"] == 0.9
        assert [r["rerank_score"] for r in record["queries"][0]["recommended"]] == [0.9, 0.2, 0.05]
        assert "queries emptied" in run.stdout

    def test_runs_the_pipeline_with_the_gate_disabled(self, tmp_path):
        run = _Run(tmp_path, self._entries(), movies=self._scored_movies())
        run("--gate-report", "--intent", "oracle")
        assert all(call.kwargs["apply_gate"] is False for call in run.pipeline.call_args_list)

    def test_plain_eval_keeps_the_gate_on(self, tmp_path):
        run = _Run(tmp_path, self._entries())
        run("--intent", "oracle")
        assert all(call.kwargs["apply_gate"] is True for call in run.pipeline.call_args_list)

    @pytest.mark.parametrize("split", ["test", "all"])
    def test_refuses_anything_but_dev(self, tmp_path, split):
        with pytest.raises(CommandError, match="dev"):
            _Run(tmp_path, self._entries())("--gate-report", "--split", split)

    def test_cannot_combine_with_sweep(self, tmp_path):
        with pytest.raises(CommandError, match="sweep"):
            _Run(tmp_path, self._entries())("--gate-report", "--sweep")

    def test_warns_when_the_reranker_produced_no_scores(self, tmp_path):
        run = _Run(tmp_path, self._entries())  # movies carry no rerank_score
        record = run("--gate-report", "--intent", "oracle")
        assert record["gate_report"]["queries_considered"] == 0
        assert "no rerank scores" in run.stdout.lower()

    def test_does_not_touch_params_yaml(self, tmp_path):
        from django.conf import settings

        before = settings.PARAMS_FILE.read_bytes()
        _Run(tmp_path, self._entries(), movies=self._scored_movies())("--gate-report", "--intent", "oracle")
        assert settings.PARAMS_FILE.read_bytes() == before



class TestLabelExport:
    """--label-export: dev-split CSV of reranked results for human labelling."""

    HEADER = [
        "query_id", "query", "context", "ranked_query", "rank", "tmdb_id", "title",
        "year", "overview", "rerank_score", "golden_hit", "label",
    ]

    def _entries(self):
        return [_entry("d1", "dev", context=["earlier"], oracle_standalone_query="standalone d1"), _entry("t1", "test")]

    @staticmethod
    def _scored_movies():
        movies = [_movie(1), _movie(2)]
        for movie, score in zip(movies, (0.9, 0.2)):
            movie["rerank_score"] = score
        return movies

    def test_writes_csv_and_record(self, tmp_path):
        import csv

        csv_path = tmp_path / "out" / "label.csv"
        run = _Run(tmp_path, self._entries(), movies=self._scored_movies(), grades={1: 2})
        record = run("--label-export", str(csv_path), "--intent", "oracle", "--k", "2")

        assert record["mode"] == "label_export"
        assert record["split"] == "dev"
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert list(rows[0]) == self.HEADER
        assert [(r["query_id"], r["rank"], r["tmdb_id"], r["golden_hit"]) for r in rows] == [
            ("d1", "1", "1", "1"), ("d1", "2", "2", "0"),
        ]
        assert rows[0]["context"] == "earlier" and rows[0]["ranked_query"] == "standalone d1"
        assert rows[0]["query"] == "query d1" and rows[0]["overview"] == "d"
        assert rows[0]["year"] == "2000" and rows[0]["label"] == ""
        assert str(csv_path) in run.stdout

    def test_runs_the_pipeline_with_the_gate_disabled(self, tmp_path):
        run = _Run(tmp_path, self._entries(), movies=self._scored_movies())
        run("--label-export", str(tmp_path / "l.csv"), "--intent", "oracle")
        assert all(call.kwargs["apply_gate"] is False for call in run.pipeline.call_args_list)

    @pytest.mark.parametrize("split", ["test", "all"])
    def test_refuses_anything_but_dev(self, tmp_path, split):
        with pytest.raises(CommandError, match="dev"):
            _Run(tmp_path, self._entries())("--label-export", str(tmp_path / "l.csv"), "--split", split)

    @pytest.mark.parametrize("other", [["--sweep"], ["--gate-report"]])
    def test_mutually_exclusive(self, tmp_path, other):
        with pytest.raises(CommandError, match="separate"):
            _Run(tmp_path, self._entries())("--label-export", str(tmp_path / "l.csv"), *other)


class TestEmptyResults:
    def test_counts_queries_the_gate_left_empty(self, tmp_path):
        run = _Run(tmp_path, [_entry()], movies=[])
        record = run("--intent", "oracle")
        assert record["counters"]["empty_results"] == 1
        assert record["queries"][0]["recommended"] == []
        assert record["queries"][0]["pool_max_rerank_score"] is None


class TestScoringWeights:
    """Swept and --weights values must sum to 1.0 so a winner can go into params.yaml."""

    @pytest.mark.parametrize("pair", [(0.8, 0.2), (0.3, 0.7), (0.5, 0.5), (3, 1)])
    def test_sums_to_one_with_popularity_held(self, pair, settings):
        from movies.management.commands.evaluate_scoring import _scoring_weights
        from recommender.settings import _validate_score_weights
        weights = _scoring_weights(*pair)
        assert set(weights) == {"semantic", "metadata", "popularity"}
        assert weights["popularity"] == settings.SCORE_WEIGHTS["popularity"]
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)
        _validate_score_weights(weights)

    def test_keeps_the_proportions(self):
        from movies.management.commands.evaluate_scoring import _scoring_weights
        weights = _scoring_weights(0.6, 0.2)
        assert weights["semantic"] / weights["metadata"] == pytest.approx(3.0, rel=1e-3)


class TestSessionAblationFlagIsGone:
    def test_session_ablation_flag_is_gone(self, tmp_path):
        with pytest.raises(CommandError):
            _Run(tmp_path, [_entry()])("--session-ablation")


class TestRunRecord:
    def test_contains_provenance_and_counters(self, tmp_path, settings):
        run = _Run(tmp_path, [_entry()])
        record = run("--k", "2", "--weights", "0.6,0.4")

        assert len(record["params_sha256"]) == 64
        assert set(record["git"]) == {"commit", "dirty"}
        assert record["k"] == 2
        assert record["options"]["weights"] == "0.6,0.4"
        pop = settings.SCORE_WEIGHTS["popularity"]
        assert record["weights"]["semantic"] == pytest.approx(0.6 * (1 - pop), abs=1e-4)
        assert record["weights"]["popularity"] == pop
        assert sum(record["weights"].values()) == pytest.approx(1.0, abs=1e-9)
        assert record["timestamp"]
        assert set(record["counters"]) >= {"unresolved_refs", "judge_failures", "misclassified"}
        query = record["queries"][0]
        assert query["recommended"][0] == {
            "tmdb_id": 1, "title": "Film 1", "release_date": "2000-01-01",
            "rerank_score": None, "hit": True, "overview": "d",
        }
        assert "ndcg_at_k" in record["aggregate"]
        assert record["aggregate"]["coverage"] == 2 / 100

    def test_default_output_is_timestamped_under_eval_runs(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        golden = _write_golden(tmp_path, [_entry()])
        movies = [_movie(1)]
        with patch(f"{MODULE}._catalog_stats", return_value=({}, 10)), \
             patch(f"{MODULE}.resolve_references", _resolver()), \
             patch(f"{MODULE}.encode_query", side_effect=_vec), \
             patch(f"{MODULE}.aclassify_and_parse", AsyncMock(return_value=MessageIntent(category="new_search"))), \
             patch(f"{MODULE}.generate_and_score", return_value=(movies, movies)), \
             patch(f"{MODULE}.ais_available", AsyncMock(return_value=True)):
            call_command("evaluate_scoring", "--golden-set", str(golden), stdout=StringIO())
        written = list((tmp_path / "eval_runs").glob("*.json"))
        assert len(written) == 1
        assert json.loads(written[0].read_text())["mode"] == "eval"

    def test_bad_weights_rejected(self, tmp_path):
        with pytest.raises(CommandError, match="--weights needs two numbers: semantic,metadata"):
            _Run(tmp_path, [_entry()])("--weights", "0.5,0.3,0.2")


class TestGitState:
    def test_git_missing_records_null(self):
        with patch(f"{MODULE}.subprocess.run", side_effect=FileNotFoundError("git")):
            assert cmd._git_state() == {"commit": None, "dirty": None}

    def test_not_a_repo_records_null(self):
        error = subprocess.CalledProcessError(128, ["git"], stderr="not a git repository")
        with patch(f"{MODULE}.subprocess.run", side_effect=error):
            assert cmd._git_state() == {"commit": None, "dirty": None}

    def test_reads_sha_and_dirty_flag(self):
        results = [MagicMock(stdout="abc123\n"), MagicMock(stdout=" M params.yaml\n")]
        with patch(f"{MODULE}.subprocess.run", side_effect=results):
            assert cmd._git_state() == {"commit": "abc123", "dirty": True}
