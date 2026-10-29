"""Tests for core.label_queries (pure, no DB): picking help-me-find-a-film
requests out of Reddit-Movie rows and linking the films the replies
recommended to TMDB ids."""

import csv

import pytest

from core.label_queries import imdb_to_tmdb, select_queries

LONG_OK = "I loved dark, gritty, inspirational dramas lately. What else should I watch tonight?"

IMDB_MAP = {f"tt{n:07d}": 1000 + n for n in range(1, 40)}


def _ids(*numbers):
    return [f"tt{n:07d}" for n in numbers]


def _thread(thread, text, replies, paths=1):
    """Rows of one Reddit thread: per path a seeker post and one reply per
    recommended group. Mirrors the documented columns (conv_id carries the
    path index, processed/raw are stringified [role, text] lists)."""
    rows = []
    for path in range(paths):
        conv_id = f"t3_{thread}_{path}/{paths}"
        rows.append(
            {
                "conv_id": conv_id,
                "turn_id": f"t3_{thread}",
                "turn_order": "0",
                "is_seeker": "True",
                "processed": repr(["USER", "see tt0000099 " + text]),
                "raw": repr(["USER", text]),
            }
        )
        for i, group in enumerate(replies, start=1):
            rows.append(
                {
                    "conv_id": conv_id,
                    "turn_id": f"t1_{thread}r{i}",
                    "turn_order": str(i),
                    "is_seeker": "False",
                    "processed": repr(["SYSTEM", "Try " + " and ".join(group)]),
                    "raw": repr(["SYSTEM", "titles"]),
                }
            )
    return rows


def fixture_rows():
    """Six threads: three usable and one each too short, too long, under-linked."""
    rows = []
    rows += _thread("aaa", LONG_OK, [_ids(1, 2), _ids(3)])
    rows += _thread("bbb", "too short", [_ids(4, 5, 6)])
    rows += _thread("ccc", "x" * 700, [_ids(7, 8, 9)])
    rows += _thread("ddd", LONG_OK + " ddd", [_ids(10, 11), ["tt9999999"]])  # 2 linked
    rows += _thread("eee", LONG_OK + " eee", [_ids(12, 13, 14, 15)], paths=3)
    rows += _thread("fff", LONG_OK + " fff", [_ids(16), _ids(17, 18)])
    return rows


KW = dict(
    count=10, test_share=0.33, seed=7, min_chars=30, max_chars=600, min_linked_recs=3
)


def _select(rows=None, **overrides):
    return select_queries(rows if rows is not None else fixture_rows(), IMDB_MAP, **{**KW, **overrides})


def _pool(n):
    rows = []
    for i in range(n):
        rows += _thread(f"p{i:03d}", f"{LONG_OK} number {i}", [_ids(1, 2, 3)])
    return rows


class TestFiltering:
    def test_length_bounds_respected(self):
        out = _select()
        assert out
        assert all(30 <= len(r["query"]) <= 600 for r in out)
        assert {r["id"] for r in out} == {"t3_aaa", "t3_eee", "t3_fff"}

    def test_bounds_come_from_arguments(self):
        out = _select(min_chars=1, max_chars=20, min_linked_recs=1)
        assert {r["id"] for r in out} == {"t3_bbb"}

    def test_records_below_min_linked_recs_dropped(self):
        ids = {r["id"] for r in _select()}
        assert "t3_ddd" not in ids  # 2 catalog films + 1 unknown id
        assert {r["id"] for r in _select(min_linked_recs=2)} >= {"t3_ddd"}

    def test_recs_are_catalog_tmdb_ids_deduped_across_paths(self):
        by_id = {r["id"]: r for r in _select()}
        assert by_id["t3_eee"]["community_recs"] == [1012, 1013, 1014, 1015]
        assert by_id["t3_aaa"]["community_recs"] == [1001, 1002, 1003]

    def test_seeker_turn_ids_are_not_counted_as_recs(self):
        # the seeker post carries tt0000099 in `processed`; it must not leak in
        imap = {**IMDB_MAP, "tt0000099": 99}
        out = select_queries(fixture_rows(), imap, **KW)
        assert all(99 not in r["community_recs"] for r in out)

    def test_query_uses_raw_text_without_role_wrapper(self):
        by_id = {r["id"]: r for r in _select()}
        assert by_id["t3_aaa"]["query"] == LONG_OK

    def test_plain_role_prefix_is_stripped(self):
        rows = _thread("zzz", "x", [_ids(1, 2, 3)])
        rows[0]["raw"] = "SEEKER: " + LONG_OK
        out = select_queries(rows, IMDB_MAP, **KW)
        assert out[0]["query"] == LONG_OK

    def test_boolean_is_seeker_accepted(self):
        rows = fixture_rows()
        for row in rows:
            row["is_seeker"] = row["is_seeker"] == "True"
        assert {r["id"] for r in _select(rows)} == {"t3_aaa", "t3_eee", "t3_fff"}

    def test_one_query_per_thread(self):
        out = _select()
        assert len({r["id"] for r in out}) == len(out)


class TestSelection:
    def test_deterministic_for_a_fixed_seed(self):
        rows = _pool(40)
        assert _select(rows, count=15) == _select(rows, count=15)

    def test_independent_of_row_order(self):
        rows = _pool(40)
        assert _select(rows, count=15) == _select(list(reversed(rows)), count=15)

    def test_seed_changes_the_draw(self):
        rows = _pool(40)
        assert {r["id"] for r in _select(rows, count=15, seed=1)} != {
            r["id"] for r in _select(rows, count=15, seed=2)
        }

    def test_count_caps_the_output(self):
        assert len(_select(_pool(40), count=15)) == 15

    def test_fewer_candidates_than_count_returns_all(self):
        assert len(_select(_pool(5), count=15)) == 5

    @pytest.mark.parametrize("count,share", [(15, 0.33), (12, 0.25), (9, 0.5), (20, 0.33)])
    def test_split_share_within_one_query(self, count, share):
        out = _select(_pool(40), count=count, test_share=share)
        n_test = sum(r["split"] == "test" for r in out)
        assert abs(n_test - share * len(out)) <= 1
        assert {r["split"] for r in out} <= {"dev", "test"}

    def test_ids_stable_when_unrelated_threads_are_added(self):
        rows = _pool(40)
        small = _select(rows, count=100)
        bigger = _select(rows + _thread("zzz", LONG_OK + " zzz", [_ids(1, 2, 3)]), count=100)
        assert {r["id"] for r in small} <= {r["id"] for r in bigger}

    def test_output_shape(self):
        for r in _select():
            assert set(r) == {"id", "query", "split", "community_recs"}
            assert isinstance(r["id"], str) and isinstance(r["query"], str)
            assert all(isinstance(t, int) for t in r["community_recs"])

    def test_output_sorted_by_id(self):
        out = _select(_pool(40), count=15)
        assert [r["id"] for r in out] == sorted(r["id"] for r in out)


class TestImdbToTmdb:
    def _write(self, tmp_path, rows):
        path = tmp_path / "catalog.csv"
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "title", "imdb_id", "vote_count"])
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_maps_imdb_to_tmdb_skipping_blanks(self, tmp_path):
        path = self._write(
            tmp_path,
            [
                {"id": "10", "title": "A", "imdb_id": "tt0000010", "vote_count": "5"},
                {"id": "11", "title": "B", "imdb_id": "", "vote_count": "5"},
                {"id": "x", "title": "C", "imdb_id": "tt0000012", "vote_count": "5"},
            ],
        )
        assert imdb_to_tmdb(path) == {"tt0000010": 10}

    def test_row_filter_limits_the_map(self, tmp_path):
        path = self._write(
            tmp_path,
            [
                {"id": "10", "title": "A", "imdb_id": "tt0000010", "vote_count": "500"},
                {"id": "11", "title": "B", "imdb_id": "tt0000011", "vote_count": "1"},
            ],
        )
        out = imdb_to_tmdb(path, row_filter=lambda row: int(row["vote_count"]) >= 100)
        assert out == {"tt0000010": 10}
