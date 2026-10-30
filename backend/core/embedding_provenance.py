"""Provenance of the vectors in Movie.embedding, and the decisions made from it.

Vectors from a different embedding model, or built from a different movie text,
have the same width as today's but are not comparable with query vectors, and
nothing in the data shows it. The EmbeddingIndex row records what built the
index; this module compares that record with the current config and holds the
pure decisions (no DB, no model) the two management commands act on.

To record one more thing about the index, add its key to current_provenance()
and the matching column to EmbeddingIndex -- comparison, messages and the
commands compare the dict. Origin differences require a read-only compatibility
sample; hard model, dimensions and text differences still require a rebuild.
"""

from dataclasses import dataclass, field

import numpy as np
from django.conf import settings

from core.embedding_service import EMBEDDING_TEXT_VERSION

_FIX_ALL = "`manage.py generate_embeddings --all` re-embeds the whole catalog and refreshes the record"


class ProvenanceError(Exception):
    """An embedding run the recorded provenance forbids; the message says what to do."""


# Recorded fields where null means "not pinned / not known" rather than a value
# of its own. A difference with a null on either side cannot be told apart from
# "same weights, recorded before pinning existed", so it is reported as a note
# and not a mismatch; two set, different values are a mismatch like any other.
_PIN_KEYS = frozenset({"revision"})


@dataclass(frozen=True)
class IndexCheck:
    status: str  # "ok" | "missing" | "mismatch" | "backend_changed"
    differences: list[str] = field(default_factory=list)
    # Differences that cannot be judged (see _PIN_KEYS); never change `status`.
    notes: list[str] = field(default_factory=list)
    origin_differences: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Verdict:
    healthy: bool
    level: str  # "ok" | "notice" | "warning" | "error"
    message: str


def current_provenance() -> dict:
    """What an index built right now would be recorded as."""
    config = settings.EFFECTIVE_PARAMS["embedding"]
    return {
        "provider": config["provider"],
        "api_provider": config.get("api_provider"),
        "model_name": settings.EMBEDDING_MODEL,
        "dimensions": settings.EMBEDDING_DIMENSIONS,
        "revision": None if config["provider"] != "local" else settings.EMBEDDING_REVISION,
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
    """Separate origin changes from hard differences and absent provenance.

    A differing _PIN_KEYS field with a null on either side goes to `notes`
    instead, so an index recorded before revisions were pinned stays ok.
    """
    if stored is None:
        return IndexCheck("missing")
    stored_flat, current_flat = _flatten(stored), _flatten(current)
    differences, notes, origin_differences = [], [], []
    for key in sorted(stored_flat.keys() | current_flat.keys()):
        was, now = stored_flat.get(key), current_flat.get(key)
        if was == now:
            continue
        line = f"{key}: index was built with {was!r}, config now has {now!r}"
        if key in {"provider", "api_provider"}:
            origin_differences.append(line)
        elif key in _PIN_KEYS and (was is None or now is None):
            notes.append(line)
        else:
            differences.append(line)
    status = "mismatch" if differences else "backend_changed" if origin_differences else "ok"
    return IndexCheck(status, differences, notes, origin_differences)


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
    if check.status == "backend_changed" and has_embeddings:
        raise ProvenanceError(
            "The embedding origin changed. A read compatibility check does not authorize "
            "mixing origins during generation:\n"
            f"{_bullets(check.origin_differences)}\n"
            f"Use the original inference mode, or {_FIX_ALL}."
        )
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
    """Report hard mismatches and origins requiring a compatibility sample."""
    if check.status == "backend_changed":
        return Verdict(
            not has_embeddings, "notice" if not has_embeddings else "error",
            "No embeddings exist; initial generation is allowed." if not has_embeddings else
            "The embedding origin changed; a sample compatibility check is required. "
            "Use the original inference mode, or explicitly run `manage.py generate_embeddings --all`.",
        )
    if check.status == "ok" and check.notes:
        return Verdict(
            True, "warning",
            "Embedding index matches the current config, but a pinned value cannot be "
            "compared because one side is not pinned:\n"
            f"{_bullets(check.notes)}\n"
            "If the vectors were built from the configured revision, nothing is wrong; "
            f"to record it, {_FIX_ALL}.",
        )
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


def validate_compatibility_vectors(stored_vectors, active_vectors, min_cosine: float) -> None:
    """Require every sampled pair to be finite, nonzero and compatible."""
    try:
        stored = np.asarray(stored_vectors, dtype=float)
        active = np.asarray(active_vectors, dtype=float)
        if stored.ndim != 2 or stored.shape != active.shape or not len(stored):
            raise ValueError
        if not np.isfinite(stored).all() or not np.isfinite(active).all():
            raise ValueError
        # Normalize separately to avoid overflowing a product of norms.
        stored_norms = np.linalg.norm(stored, axis=1)
        active_norms = np.linalg.norm(active, axis=1)
        if (not np.isfinite(stored_norms).all() or not np.isfinite(active_norms).all()
                or np.any(stored_norms == 0) or np.any(active_norms == 0)):
            raise ValueError
        similarities = np.sum((stored / stored_norms[:, None]) * (active / active_norms[:, None]), axis=1)
        if np.any(similarities < min_cosine):
            raise ValueError
    except (ValueError, TypeError, OverflowError) as exc:
        raise ProvenanceError(
            "Embedding compatibility could not be confirmed for every sampled movie. "
            "Use the original inference mode, or explicitly run "
            "`manage.py generate_embeddings --all` to rebuild the index."
        ) from exc
