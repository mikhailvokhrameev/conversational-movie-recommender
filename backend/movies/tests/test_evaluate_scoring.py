"""Tests for the evaluate_scoring management command.

No DB and no models: the catalog stats, golden-reference resolution, encoder,
LLM, and pipeline are patched at the command module, so these exercise the
command's own logic -- intent modes, session vectors, relevance, splits,
counters, and the run record."""

import json
import subprocess
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from core.ollama_client import MessageIntent
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
            category="new_search", semantic_query="rewritten query", genres=["Drama"],
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
        ).model_dump(exclude={"category", "semantic_query"})
        assert args[2] is None
        assert kwargs["query_text"] == "query q1"
        assert record["intent_mode"] == "e2e"
        assert record["queries"][0]["predicted_category"] == "new_search"

    def test_context_becomes_user_history(self, tmp_path):
        run = _Run(tmp_path, [_entry(context=["first", "second"])])
        run()
        run.classify.assert_awaited_once_with(
            "query q1", [{"role": "user", "content": "first"}, {"role": "user", "content": "second"}],
        )

    @pytest.mark.parametrize("category", ["follow_up", "general_chat"])
    def test_non_search_category_still_searches_and_counts_misclassified(self, tmp_path, category):
        run = _Run(tmp_path, [_entry()], intent=MessageIntent(category=category, semantic_query="x"))
        record = run()
        run.pipeline.assert_called_once()
        assert record["counters"]["misclassified"] == 1
        assert record["queries"][0]["misclassified"] is True


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
        fallback = MessageIntent(category="new_search", semantic_query="query q1")
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

    def test_missing_oracle_intent_is_empty(self, tmp_path):
        run = _Run(tmp_path, [_entry()])
        run("--intent", "oracle")
        assert run.pipeline.call_args.args[1] == {}


class TestSessionVector:
    def test_no_context_means_no_session_vector(self, tmp_path):
        run = _Run(tmp_path, [_entry()])
        run("--intent", "oracle")
        assert run.pipeline.call_args.args[2] is None

    def test_context_folds_like_production(self, tmp_path):
        from core.session_manager import update_preference_vector, vector_for_scoring

        run = _Run(tmp_path, [_entry(context=["a sci-fi movie", "with robots"], category="refinement")])
        run("--intent", "oracle")

        expected = update_preference_vector(None, _vec("a sci-fi movie"))
        expected = update_preference_vector(expected, _vec("with robots"))
        expected = vector_for_scoring(expected, _vec("query q1"), "refinement")
        assert run.pipeline.call_args.args[2] == pytest.approx(expected)

    def test_e2e_uses_predicted_category_for_scoring_vector(self, tmp_path):
        from core.session_manager import update_preference_vector, vector_for_scoring

        intent = MessageIntent(category="refinement", semantic_query="funnier sci-fi")
        run = _Run(tmp_path, [_entry(context=["a sci-fi movie"], category="new_search")], intent=intent)
        run()

        prior = update_preference_vector(None, _vec("a sci-fi movie"))
        expected = vector_for_scoring(prior, _vec("funnier sci-fi"), "refinement")
        assert run.pipeline.call_args.args[2] == pytest.approx(expected)


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
        run = _Run(tmp_path, [_entry(context=["a sci-fi movie"])], judge=[3, 3])
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


class TestRunRecord:
    def test_contains_provenance_and_counters(self, tmp_path, settings):
        run = _Run(tmp_path, [_entry()])
        record = run("--k", "2", "--weights", "0.5,0.3,0.2")

        assert len(record["params_sha256"]) == 64
        assert set(record["git"]) == {"commit", "dirty"}
        assert record["k"] == 2
        assert record["options"]["weights"] == "0.5,0.3,0.2"
        assert record["weights"]["semantic"] == 0.5
        assert record["weights"]["popularity"] == settings.SCORE_WEIGHTS["popularity"]
        assert record["timestamp"]
        assert set(record["counters"]) >= {"unresolved_refs", "judge_failures", "misclassified"}
        query = record["queries"][0]
        assert query["recommended"][0] == {"tmdb_id": 1, "title": "Film 1", "release_date": "2000-01-01"}
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
        with pytest.raises(CommandError, match="weights"):
            _Run(tmp_path, [_entry()])("--weights", "0.5,0.5")


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
