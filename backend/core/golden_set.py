"""Golden set for offline evaluation: loading, validation, and resolving its
movie references against the catalog.

The golden set (`data/golden_set.json`) labels each query with the movies that
answer it. Relevance is membership in that list -- not genre overlap, which the
`metadata` scoring signal computes from the same genres and so would grade the
ranker against its own input.

References are either `{"tmdb_id": int}` or `{"title": str, "year": int}`,
optionally with a `grade` (1 relevant, 2 highly relevant). Title references
exist because labels can be written without catalog access; they are resolved
here at eval time, and anything that does not resolve is returned to the
caller to report -- an unresolved reference is a labelling or catalog gap that
must stay visible, not a smaller denominator nobody notices.

Kept free of command-line concerns so other evaluations (e.g. calibrating a
relevance threshold) can reuse the loader and resolver.
"""

import json
import logging
from pathlib import Path

from django.conf import settings
from django.db.models import Q

from movies.models import Movie

logger = logging.getLogger(__name__)

SPLITS = ("dev", "test")
GRADES = (1, 2)
DEFAULT_GRADE = 1
# Categories an oracle-mode entry may claim for its final message: the two the
# chat view routes through the retrieval pipeline.
SEARCH_CATEGORIES = ("new_search", "refinement")


class GoldenSetError(ValueError):
    pass


def default_golden_set_path() -> Path:
    """data/golden_set.json, found the way settings finds params.yaml: the
    container location first (data/ is mounted at /app/data), then the repo
    root for running outside Docker."""
    candidates = [
        Path(settings.BASE_DIR) / "data" / "golden_set.json",
        Path(settings.BASE_DIR).parent / "data" / "golden_set.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[-1]


def load_golden_set(path) -> list[dict]:
    """Load and validate the golden set; returns its query entries."""
    with open(path) as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("queries"), list):
        raise GoldenSetError(f"{path}: expected an object with a 'queries' list")
    validate_entries(data["queries"])
    return data["queries"]


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_str_list(value) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) and v.strip() for v in value)


def _reference_problems(ref) -> list[str]:
    if not isinstance(ref, dict):
        return [f"reference {ref!r} is not an object"]
    problems = []
    if "tmdb_id" in ref:
        if not _is_int(ref["tmdb_id"]):
            problems.append(f"reference {ref!r}: tmdb_id must be an integer")
    elif not (isinstance(ref.get("title"), str) and ref["title"].strip() and _is_int(ref.get("year"))):
        problems.append(f"reference {ref!r}: needs tmdb_id, or title and an integer year")
    if "grade" in ref and ref["grade"] not in GRADES:
        problems.append(f"reference {ref!r}: grade must be one of {GRADES}")
    return problems


def _entry_problems(entry) -> list[str]:
    if not isinstance(entry, dict):
        return [f"entry {entry!r} is not an object"]
    problems = []
    if not (isinstance(entry.get("id"), str) and entry["id"].strip()):
        problems.append("missing a string id")
    if not (isinstance(entry.get("query"), str) and entry["query"].strip()):
        problems.append("missing a query")
    if entry.get("split") not in SPLITS:
        problems.append(f"split must be one of {SPLITS} (got {entry.get('split')!r})")
    relevant = entry.get("relevant")
    if not isinstance(relevant, list) or not relevant:
        problems.append("needs at least one relevant reference")
    else:
        for ref in relevant:
            problems.extend(_reference_problems(ref))
    if "negated_genres" in entry and not _is_str_list(entry["negated_genres"]):
        problems.append("negated_genres must be a list of strings")
    if "oracle_intent" in entry and not isinstance(entry["oracle_intent"], dict):
        problems.append("oracle_intent must be an object")
    if "context" in entry and not _is_str_list(entry["context"]):
        problems.append("context must be a list of prior user messages")
    if "category" in entry and entry["category"] not in SEARCH_CATEGORIES:
        problems.append(f"category must be one of {SEARCH_CATEGORIES}")
    return problems


def validate_entries(entries: list) -> None:
    """Raise GoldenSetError listing every problem in the entries, if any."""
    problems = []
    seen_ids = set()
    for index, entry in enumerate(entries):
        label = entry.get("id", f"#{index}") if isinstance(entry, dict) else f"#{index}"
        problems.extend(f"{label}: {p}" for p in _entry_problems(entry))
        if isinstance(entry, dict) and "id" in entry:
            if entry["id"] in seen_ids:
                problems.append(f"{label}: duplicate id")
            seen_ids.add(entry["id"])
    if problems:
        raise GoldenSetError("invalid golden set:\n  " + "\n  ".join(problems))


def select_split(entries: list[dict], split: str) -> list[dict]:
    """Entries of one split, or every entry for "all"."""
    if split == "all":
        return list(entries)
    if split not in SPLITS:
        raise GoldenSetError(f"unknown split {split!r}")
    return [entry for entry in entries if entry["split"] == split]


def reference_label(ref: dict) -> str:
    if "tmdb_id" in ref:
        return f"tmdb:{ref['tmdb_id']}"
    return f"{ref['title']} ({ref['year']})"


def _resolve_one(ref: dict) -> int | None:
    if "tmdb_id" in ref:
        return ref["tmdb_id"] if Movie.objects.filter(tmdb_id=ref["tmdb_id"]).exists() else None
    title = ref["title"].strip()
    matches = list(
        Movie.objects.filter(
            Q(serial_name__iexact=title) | Q(original_title__iexact=title),
            release_date__year=ref["year"],
        )
        .order_by("-vote_count")
        .values_list("tmdb_id", flat=True)[:2]
    )
    if len(matches) > 1:
        logger.warning(
            f"Golden reference {reference_label(ref)} matches several movies; "
            f"using the most-voted one (tmdb:{matches[0]})"
        )
    return matches[0] if matches else None


def resolve_references(refs: list[dict]) -> tuple[dict[int, int], list[dict]]:
    """Resolve golden references to catalog tmdb_ids.

    Returns (grades, unresolved): `grades` maps each resolved tmdb_id to its
    grade (the highest, if two references land on the same movie);
    `unresolved` lists the references with no catalog match, for the caller
    to report and count.
    """
    grades: dict[int, int] = {}
    unresolved: list[dict] = []
    for ref in refs:
        tmdb_id = _resolve_one(ref)
        if tmdb_id is None:
            unresolved.append(ref)
            continue
        grades[tmdb_id] = max(grades.get(tmdb_id, 0), ref.get("grade", DEFAULT_GRADE))
    return grades, unresolved
