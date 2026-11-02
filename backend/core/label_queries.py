"""Label-query set: movie requests mined from Reddit-Movie, ready for hand grading.

Reddit-Movie (He et al. 2023, `ZhankuiHe/reddit_movie_large_v1`) holds
r/movies-style recommendation threads. One CSV row is one turn: `conv_id`
("t3_<thread>_<path>/<paths>", one conversation per path through the reply
tree), `turn_order` (0 = the original post), `is_seeker`, and the turn text in
`processed` (movie mentions replaced by IMDb ids such as `tt0053779`) and `raw`
(titles kept). Both text columns are stringified `[role, text]` lists.

The seeker's original post (its `raw` text, so the request reads as written) is
the query; the films the replies recommended (IMDb ids found in the non-seeker
rows' `processed` text) are mapped to TMDB ids through the catalog and become
`community_recs`, a *starting list* for the human grader, not a label.

Pure functions, no database. Selection is a deterministic function of the seed
and the thread ids: each candidate is ranked by a hash of (seed, id), so the
same input gives the same set, row order is irrelevant, and adding unrelated
threads rarely changes which of the others are picked.
"""

import ast
import csv
import hashlib
import re
from pathlib import Path
from typing import Callable

_IMDB_ID = re.compile(r"\btt\d{7,}\b")
_CONV_ID = re.compile(r"^(?P<thread>.+)_(?P<path>\d+)/(?P<paths>\d+)$")
_ROLE_PREFIX = re.compile(r"^\s*(?:USER|SYSTEM|SEEKER|RECOMMENDER)\s*:\s*", re.IGNORECASE)

# The catalog CSV has fields up to several hundred KB (see import_catalog).
csv.field_size_limit(10_000_000)


def imdb_to_tmdb(
    catalog_csv: Path, row_filter: Callable[[dict], bool] | None = None
) -> dict[str, int]:
    """Map IMDb id -> TMDB id from the `imdb_id` and `id` columns of the TMDB
    catalog CSV. Rows with a blank imdb_id or a non-numeric id are skipped.
    `row_filter(row)` lets the caller keep only rows the imported catalog would
    keep (the quality filter), so recs are films the app can actually show."""
    mapping: dict[str, int] = {}
    with open(catalog_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            imdb_id = (row.get("imdb_id") or "").strip()
            if not imdb_id:
                continue
            try:
                tmdb_id = int(row["id"])
            except (KeyError, TypeError, ValueError):
                continue
            if row_filter is not None and not row_filter(row):
                continue
            mapping[imdb_id] = tmdb_id
    return mapping


def _turn_text(value) -> str:
    """The text of a `[role, text]` cell, tolerating a plain "ROLE: text"."""
    if not isinstance(value, str):
        return ""
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        parsed = None
    if isinstance(parsed, (list, tuple)) and parsed:
        return str(parsed[-1]).strip()
    return _ROLE_PREFIX.sub("", value, count=1).strip()


def _is_true(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1"}
    return bool(value)


def _int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _thread_id(conv_id: str) -> str:
    match = _CONV_ID.match(conv_id)
    return match.group("thread") if match else conv_id


def _group_threads(records: list[dict], imdb_map: dict[str, int]) -> dict[str, dict]:
    """thread id -> {"query": str, "recs": [tmdb_id]} (recs in reply order, deduped)."""
    threads: dict[str, dict] = {}
    for row in records:
        conv_id = row.get("conv_id") or ""
        if not conv_id:
            continue
        thread = threads.setdefault(_thread_id(conv_id), {"query": None, "found": [], "seeker_ids": set()})
        order = _int(row.get("turn_order"))
        if _is_true(row.get("is_seeker")):
            thread["seeker_ids"].update(_IMDB_ID.findall(_turn_text(row.get("processed"))))
            if order == 0 and thread["query"] is None:
                thread["query"] = _turn_text(row.get("raw"))
            continue
        text = _turn_text(row.get("processed"))
        for position, imdb_id in enumerate(_IMDB_ID.findall(text)):
            thread["found"].append((order, row.get("turn_id") or "", position, imdb_id))

    for thread in threads.values():
        recs: list[int] = []
        for _, _, _, imdb_id in sorted(thread.pop("found")):
            if imdb_id in thread["seeker_ids"]:
                continue
            tmdb_id = imdb_map.get(imdb_id)
            if tmdb_id is not None and tmdb_id not in recs:
                recs.append(tmdb_id)
        thread["recs"] = recs
    return threads


def _rank(seed: int, thread_id: str) -> str:
    return hashlib.sha256(f"{seed}:{thread_id}".encode()).hexdigest()


def select_queries(
    records: list[dict],
    imdb_map: dict[str, int],
    count: int,
    test_share: float,
    seed: int,
    min_chars: int,
    max_chars: int,
    min_linked_recs: int,
    existing_splits: dict[str, str] | None = None,
) -> list[dict]:
    """Pick up to `count` usable requests and split them dev/test.

    A thread is usable when its seeker post is between `min_chars` and
    `max_chars` long and its replies recommend at least `min_linked_recs`
    distinct catalog films. Returns `{"id", "query", "split", "community_recs"}`
    records sorted by id, where id is the Reddit thread id.
    """
    candidates = [
        (thread_id, thread)
        for thread_id, thread in _group_threads(records, imdb_map).items()
        if thread["query"]
        and min_chars <= len(thread["query"]) <= max_chars
        and len(thread["recs"]) >= min_linked_recs
    ]
    candidates.sort(key=lambda item: _rank(seed, item[0]))
    chosen = candidates[:count]
    n_test = round(len(chosen) * test_share)
    out = [
        {
            "id": thread_id,
            "query": thread["query"],
            "split": (existing_splits or {}).get(
                thread_id, "test" if index < n_test else "dev"
            ),
            "community_recs": thread["recs"],
        }
        for index, (thread_id, thread) in enumerate(chosen)
    ]
    out.sort(key=lambda record: record["id"])
    return out


def read_turns(path: Path) -> list[dict]:
    """Read the downloaded turn CSV, rejecting silent wrong-format imports."""
    required = {"conv_id", "turn_id", "turn_order", "is_seeker", "processed", "raw"}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, strict=True)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("missing CSV columns: " + ", ".join(sorted(missing)))
        rows = list(reader)
        if any(None in row or any(row[key] is None for key in required) for row in rows):
            raise ValueError("CSV row has missing or extra cells")
        return rows
