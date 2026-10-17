"""Map parsed intent values onto exact catalog values.

Genres, negations, countries and original languages are compared with the
catalog as exact strings (metadata overlap, `genres__contains`,
`country__contains`, `original_language__in`), so a near-miss name from the
LLM ("USA" for "United States of America", "sci-fi" for "Science Fiction")
silently loses or empties the filter. An EntityResolver turns each raw value
into a catalog value, in layers:

1. exact: casefold + collapsed whitespace against the catalog values;
2. synonyms from params.yaml (`intent.resolution.<field>.synonyms`), whose
   targets are checked against the catalog on first use;
3. embedding nearest catalog value, accepted only above `match_threshold`
   and at least `match_margin` ahead of the runner-up (only when the config
   has a threshold -- ISO language codes mean nothing to an embedder);
4. otherwise the value is dropped with an INFO line. A dropped filter
   widens the answer; a guessed one silently swaps one value for another.

Catalog values and their embeddings load on first use and are cached for
the process. Callers on the event loop must run `resolve` in a thread
(`sync_to_async`): the DB-backed catalogs use the ORM.
"""

import logging
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np

from core.embedding_service import encode_texts

logger = logging.getLogger(__name__)


def normalize(value: str) -> str:
    """The comparison key: casefolded, whitespace collapsed."""
    return " ".join(value.split()).casefold()


@dataclass(frozen=True)
class Resolution:
    raw: str
    value: str | None
    via: str  # "exact" | "synonym" | "embedding" | "dropped"
    top: tuple[tuple[str, float], ...] = ()


class EntityResolver:
    def __init__(self, field: str, catalog_values: Callable[[], list[str]], config: dict):
        self.field = field
        self._catalog_values = catalog_values
        self._synonyms = config.get("synonyms") or {}
        self.match_threshold = config.get("match_threshold")
        self.match_margin = config.get("match_margin")
        self._lock = threading.Lock()
        self._values: list[str] | None = None
        self._by_key: dict[str, str] = {}
        self._synonym_map: dict[str, str] = {}
        self._embeddings: np.ndarray | None = None

    @property
    def uses_embeddings(self) -> bool:
        return self.match_threshold is not None

    def resolve(self, raw: list) -> list[str]:
        """Catalog values for `raw`, deduplicated, in input order; unresolved
        values are left out."""
        resolved = []
        for result in self.resolve_each(raw):
            if result.value is not None and result.value not in resolved:
                resolved.append(result.value)
        return resolved

    def resolve_each(self, raw: list) -> list[Resolution]:
        """One Resolution per non-blank string in `raw`, saying which layer
        matched (and, for the embedding layer, the top-2 scores)."""
        texts = [r for r in raw or [] if isinstance(r, str) and r.strip()]
        if not texts:
            return []
        self._load()

        results: list[Resolution | None] = []
        pending: list[int] = []
        for i, text in enumerate(texts):
            key = normalize(text)
            if key in self._by_key:
                results.append(Resolution(text, self._by_key[key], "exact"))
            elif key in self._synonym_map:
                results.append(Resolution(text, self._synonym_map[key], "synonym"))
            else:
                results.append(None)
                pending.append(i)

        if pending and self.uses_embeddings and self._values:
            tops = self.nearest([texts[i] for i in pending])
            for i, top in zip(pending, tops):
                results[i] = self._accept(texts[i], top)
        for i in pending:
            if results[i] is None:
                logger.info(f"Entity dropped: {self.field} '{texts[i]}' (no exact or synonym match)")
                results[i] = Resolution(texts[i], None, "dropped")
        return results

    def nearest(self, texts: list[str], n: int = 2) -> list[tuple[tuple[str, float], ...]]:
        """Top-n catalog values by cosine similarity, for each text."""
        if not self.uses_embeddings:
            raise ValueError(f"{self.field}: resolution has no embedding fallback")
        self._load()
        if not self._values:
            return [() for _ in texts]
        catalog = self._catalog_embeddings()
        queries = _unit_rows(encode_texts(texts))
        scores = queries @ catalog.T
        tops = []
        for row in scores:
            order = np.argsort(-row)[:n]
            tops.append(tuple((self._values[j], float(row[j])) for j in order))
        return tops

    def _accept(self, text: str, top: tuple[tuple[str, float], ...]) -> Resolution:
        best, best_score = top[0]
        margin = best_score - top[1][1] if len(top) > 1 else None
        if best_score >= self.match_threshold and (margin is None or margin >= self.match_margin):
            logger.debug(f"Entity resolved: {self.field} '{text}' -> '{best}' ({_fmt_top(top)})")
            return Resolution(text, best, "embedding", top)
        logger.info(
            f"Entity dropped: {self.field} '{text}' (low confidence: {_fmt_top(top)}; "
            f"needs >= {self.match_threshold} and margin >= {self.match_margin})"
        )
        return Resolution(text, None, "dropped", top)

    def _load(self) -> None:
        """Load the catalog values and check the synonym targets, once. An
        empty catalog (DB not imported yet) is retried on the next call."""
        if self._values:
            return
        with self._lock:
            if self._values:
                return
            values = list(dict.fromkeys(v for v in self._catalog_values() if isinstance(v, str) and v.strip()))
            by_key = {normalize(v): v for v in values}
            synonym_map = {}
            if values:
                for name, target in self._synonyms.items():
                    canonical = by_key.get(normalize(target))
                    if canonical is None:
                        logger.warning(
                            f"params.yaml: intent.resolution.{self.field}.synonyms maps '{name}' to "
                            f"'{target}', which is not a catalog value; the synonym is ignored."
                        )
                    else:
                        synonym_map[normalize(name)] = canonical
            self._by_key, self._synonym_map = by_key, synonym_map
            self._values = values

    def _catalog_embeddings(self) -> np.ndarray:
        if self._embeddings is None:
            with self._lock:
                if self._embeddings is None:
                    self._embeddings = _unit_rows(encode_texts(self._values))
                    logger.info(f"Cached embeddings for {len(self._values)} catalog {self.field}")
        return self._embeddings


def _unit_rows(matrix) -> np.ndarray:
    """Rows scaled to unit length (zero rows stay zero), so a dot product is
    the cosine similarity."""
    matrix = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def _fmt_top(top) -> str:
    return ", ".join(f"'{name}' {score:.3f}" for name, score in top)


def catalog_countries() -> list[str]:
    """Distinct production countries across the catalog (Movie.country is a
    list per movie)."""
    from movies.models import Movie

    countries = set()
    for movie_countries in Movie.objects.values_list("country", flat=True).distinct():
        countries.update(c for c in movie_countries or [] if isinstance(c, str))
    return sorted(countries)


def catalog_languages() -> list[str]:
    """Distinct non-blank original_language codes in the catalog."""
    from movies.models import Movie

    return sorted(
        Movie.objects.exclude(original_language="")
        .values_list("original_language", flat=True)
        .distinct()
    )
