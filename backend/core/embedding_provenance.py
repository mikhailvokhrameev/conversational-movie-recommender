"""Provenance of the vectors in Movie.embedding, and the decisions made from it.

Vectors from a different embedding model, or built from a different movie text,
have the same width as today's but are not comparable with query vectors, and
nothing in the data shows it. The EmbeddingIndex row records what built the
index; this module compares that record with the current config and holds the
pure decisions (no DB, no model) the two management commands act on.

To record one more thing about the index, add its key to current_provenance()
and the matching column to EmbeddingIndex -- comparison, messages and the
commands iterate the dict and need no further change.
"""

from dataclasses import dataclass, field

from django.conf import settings

from core.embedding_service import EMBEDDING_TEXT_VERSION

_FIX_ALL = "`manage.py generate_embeddings --all` re-embeds the whole catalog and refreshes the record"


class ProvenanceError(Exception):
    """An embedding run the recorded provenance forbids; the message says what to do."""


@dataclass(frozen=True)
class IndexCheck:
    status: str  # "ok" | "missing" | "mismatch"
    differences: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Verdict:
    healthy: bool
    level: str  # "ok" | "notice" | "warning" | "error"
    message: str


def current_provenance() -> dict:
    """What an index built right now would be recorded as."""
    return {
        "model_name": settings.EMBEDDING_MODEL,
        "dimensions": settings.EMBEDDING_DIMENSIONS,
        "text_version": EMBEDDING_TEXT_VERSION,
        "text_config": {
            "description_chars": settings.EMBEDDING_TEXT_DESCRIPTION_CHARS,
            "max_keywords": settings.EMBEDDING_TEXT_MAX_KEYWORDS,
        },
    }


def _flatten(record: dict) -> dict:
    flat = {}
    for key, value in record.items():
        if isinstance(value, dict):
            for inner_key, inner_value in value.items():
                flat[f"{key}.{inner_key}"] = inner_value
        else:
            flat[key] = value
    return flat


def compare_provenance(stored: dict | None, current: dict) -> IndexCheck:
    """ok / missing (no record) / mismatch (with one line per differing field)."""
    if stored is None:
        return IndexCheck("missing")
    stored_flat, current_flat = _flatten(stored), _flatten(current)
    differences = [
        f"{key}: index was built with {stored_flat.get(key)!r}, config now has {current_flat.get(key)!r}"
        for key in sorted(stored_flat.keys() | current_flat.keys())
        if stored_flat.get(key) != current_flat.get(key)
    ]
    return IndexCheck("mismatch" if differences else "ok", differences)


def _bullets(differences: list[str]) -> str:
    return "\n".join(f"  - {line}" for line in differences)


def plan_generate(check: IndexCheck, has_embeddings: bool, rebuild_all: bool, adopt: bool) -> str:
    """Decide what generate_embeddings does: "embed_missing", "embed_all" or "adopt".

    Raises ProvenanceError for every run that would mix vector spaces or
    assert provenance nobody vouched for.
    """
    if rebuild_all and adopt:
        raise ProvenanceError("--all and --adopt are mutually exclusive: --all rebuilds, --adopt keeps the vectors.")
    if rebuild_all:
        return "embed_all"
    if adopt:
        if check.status != "missing":
            raise ProvenanceError(
                "--adopt only records a legacy index that has no provenance record yet. "
                f"This index already has one; after a deliberate config change, {_FIX_ALL}."
            )
        if not has_embeddings:
            raise ProvenanceError("--adopt has nothing to adopt: no embeddings exist yet. Run without --adopt.")
        return "adopt"
    if check.status == "mismatch":
        raise ProvenanceError(
            "The embedding index was built with a different config; embedding only the "
            "missing rows would mix incomparable vectors:\n"
            f"{_bullets(check.differences)}\n"
            f"Revert the config, or {_FIX_ALL}."
        )
    if check.status == "missing" and has_embeddings:
        raise ProvenanceError(
            "Embeddings exist but nothing records what built them (an index from before "
            "provenance tracking). Either:\n"
            "  - `--adopt` records the current config as the index's provenance without "
            "re-embedding; you assert the vectors were built with it, and\n"
            "  - `--all` re-embeds the whole catalog and records it."
        )
    return "embed_missing"


def check_verdict(check: IndexCheck, has_embeddings: bool) -> Verdict:
    """What check_embedding_index reports; only a mismatch is unhealthy."""
    if check.status == "ok":
        return Verdict(True, "ok", "Embedding index matches the current config.")
    if check.status == "missing":
        if not has_embeddings:
            return Verdict(
                True, "notice",
                "No embedding index yet: `manage.py generate_embeddings` has not run.",
            )
        return Verdict(
            True, "warning",
            "Embeddings exist but have no provenance record (index built before provenance "
            "tracking). If they were built with the current config, run "
            "`manage.py generate_embeddings --adopt` to record it.",
        )
    return Verdict(
        False, "error",
        "The embedding index does not match the current config; query vectors would not be "
        "comparable with the stored ones:\n"
        f"{_bullets(check.differences)}\n"
        "Revert the config, or run `manage.py generate_embeddings --all` to rebuild the index.",
    )
