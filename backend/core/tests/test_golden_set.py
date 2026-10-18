"""Tests for core.golden_set and the shipped data/golden_set.json.

Everything except TestResolveReferences is pure (no DB)."""

import json

import pytest

from core.golden_set import (
    GoldenSetError,
    default_golden_set_path,
    load_golden_set,
    reference_label,
    resolve_references,
    select_split,
    validate_entries,
)

GOLDEN_SET_PATH = default_golden_set_path()


def _entry(**overrides):
    entry = {
        "id": "q1",
        "query": "a comedy",
        "split": "dev",
        "relevant": [{"title": "Superbad", "year": 2007}],
    }
    entry.update(overrides)
    return entry


class TestShippedGoldenSet:
    @pytest.fixture(scope="class")
    def entries(self):
        return load_golden_set(GOLDEN_SET_PATH)

    def test_loads_and_validates(self, entries):
        assert len(entries) >= 30

    def test_every_entry_has_a_split_and_a_reference(self, entries):
        for entry in entries:
            assert entry["split"] in ("dev", "test")
            assert entry["relevant"]

    def test_ids_are_unique(self, entries):
        ids = [entry["id"] for entry in entries]
        assert len(ids) == len(set(ids))

    def test_both_splits_have_negations_and_multi_turn_entries(self, entries):
        for split in ("dev", "test"):
            subset = select_split(entries, split)
            assert any(e.get("negated_genres") for e in subset), split
            assert any(e.get("context") for e in subset), split

    def test_split_is_roughly_two_thirds_dev(self, entries):
        dev = len(select_split(entries, "dev"))
        assert 0.55 <= dev / len(entries) <= 0.75

    def test_has_at_least_five_multi_turn_entries(self, entries):
        assert sum(1 for e in entries if e.get("context")) >= 5

    def test_labels_are_title_year_references_not_invented_ids(self, entries):
        # Labels were drafted without catalog access, so a tmdb_id here would
        # be a guess. Remove this once reviewed ids are added on purpose.
        for entry in entries:
            for ref in entry["relevant"]:
                assert "tmdb_id" not in ref, (entry["id"], ref)

    def test_every_context_entry_has_a_standalone_rewrite(self, entries):
        for entry in entries:
            if entry.get("context"):
                assert entry["oracle_standalone_query"].strip(), entry["id"]

    def test_note_flags_the_labels_as_unreviewed_drafts(self):
        note = json.loads(GOLDEN_SET_PATH.read_text())["_note"].lower()
        assert "draft" in note and "review" in note


class TestValidation:
    def test_valid_entry_passes(self):
        validate_entries([_entry(), _entry(id="q2", relevant=[{"tmdb_id": 5, "grade": 2}])])

    @pytest.mark.parametrize("overrides, problem", [
        ({"split": "train"}, "split"),
        ({"relevant": []}, "relevant"),
        ({"relevant": [{"title": "Superbad"}]}, "title and an integer year"),
        ({"relevant": [{"tmdb_id": "5"}]}, "tmdb_id"),
        ({"relevant": [{"tmdb_id": 5, "grade": 3}]}, "grade"),
        ({"query": ""}, "query"),
        ({"context": "I want a sci-fi movie"}, "context"),
        ({"category": "follow_up"}, "category"),
        ({"oracle_intent": ["Comedy"]}, "oracle_intent"),
        ({"negated_genres": "Horror"}, "negated_genres"),
        ({"context": ["I want a sci-fi movie"]}, "oracle_standalone_query"),
        ({"oracle_standalone_query": ""}, "oracle_standalone_query"),
        ({"oracle_standalone_query": "  "}, "oracle_standalone_query"),
        ({"oracle_standalone_query": ["a comedy"]}, "oracle_standalone_query"),
        ({"context": ["I want a sci-fi movie"], "oracle_standalone_query": 5}, "oracle_standalone_query"),
    ])
    def test_rejects_malformed_entry(self, overrides, problem):
        with pytest.raises(GoldenSetError, match=problem):
            validate_entries([_entry(**overrides)])

    def test_standalone_query_optional_without_context(self):
        validate_entries([_entry(), _entry(id="q2", oracle_standalone_query="a comedy")])

    def test_standalone_query_accepted_with_context(self):
        validate_entries([_entry(context=["a sci-fi movie"], oracle_standalone_query="a funny sci-fi movie")])

    def test_rejects_duplicate_ids(self):
        with pytest.raises(GoldenSetError, match="duplicate id"):
            validate_entries([_entry(), _entry()])

    def test_rejects_file_without_queries_list(self, tmp_path):
        path = tmp_path / "g.json"
        path.write_text(json.dumps([_entry()]))
        with pytest.raises(GoldenSetError, match="queries"):
            load_golden_set(path)


class TestSelectSplit:
    def test_filters_by_split_or_returns_all(self):
        entries = [_entry(id="a"), _entry(id="b", split="test")]
        assert [e["id"] for e in select_split(entries, "dev")] == ["a"]
        assert [e["id"] for e in select_split(entries, "test")] == ["b"]
        assert len(select_split(entries, "all")) == 2

    def test_rejects_unknown_split(self):
        with pytest.raises(GoldenSetError):
            select_split([], "train")


def test_reference_label():
    assert reference_label({"tmdb_id": 5}) == "tmdb:5"
    assert reference_label({"title": "Up", "year": 2009}) == "Up (2009)"


@pytest.mark.django_db
class TestResolveReferences:
    @staticmethod
    def _movie(tmdb_id, title, year, original_title="", vote_count=100):
        from movies.models import Movie

        return Movie.objects.create(
            tmdb_id=tmdb_id, serial_name=title, original_title=original_title,
            release_date=f"{year}-06-01", vote_count=vote_count,
        )

    def test_resolves_title_case_insensitively_with_year(self):
        self._movie(1, "Spirited Away", 2001)
        grades, unresolved = resolve_references([{"title": "spirited away", "year": 2001, "grade": 2}])
        assert grades == {1: 2}
        assert unresolved == []

    def test_resolves_by_original_title(self):
        self._movie(2, "Rififi", 1955, original_title="Du rififi chez les hommes")
        grades, _ = resolve_references([{"title": "Du Rififi chez les hommes", "year": 1955}])
        assert grades == {2: 1}

    def test_wrong_year_is_unresolved(self):
        self._movie(3, "Ghost in the Shell", 2017)
        ref = {"title": "Ghost in the Shell", "year": 1995}
        grades, unresolved = resolve_references([ref])
        assert grades == {}
        assert unresolved == [ref]

    def test_tmdb_id_resolves_only_when_in_catalog(self):
        self._movie(4, "Up", 2009)
        grades, unresolved = resolve_references([{"tmdb_id": 4}, {"tmdb_id": 999}])
        assert grades == {4: 1}
        assert unresolved == [{"tmdb_id": 999}]

    def test_ambiguous_title_takes_most_voted(self):
        self._movie(5, "Earth", 2007, vote_count=50)
        self._movie(6, "Earth", 2007, vote_count=500)
        grades, _ = resolve_references([{"title": "Earth", "year": 2007}])
        assert grades == {6: 1}

    def test_duplicate_references_keep_highest_grade(self):
        self._movie(7, "Heat", 1995)
        grades, _ = resolve_references([{"title": "Heat", "year": 1995}, {"tmdb_id": 7, "grade": 2}])
        assert grades == {7: 2}
