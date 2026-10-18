"""EntityResolver: maps parsed intent values onto exact catalog values.

No DB and no model: catalog values are plain callables and encode_texts is
patched with a fixed vector table, so every test runs on the host."""

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.entity_resolution import EntityResolver

ENCODE = "core.entity_resolution.encode_texts"

COUNTRIES = ["United States of America", "United Kingdom", "South Korea", "North Korea", "France"]
GENRES = ["Comedy", "Horror", "Science Fiction", "Thriller", "Crime"]

EMBEDDED = {"match_threshold": 0.5, "match_margin": 0.05, "synonyms": {}}


def _encoder(table: dict[str, list[float]]):
    """encode_texts stub: each text maps to its row in `table` (unknown -> zeros)."""
    dim = len(next(iter(table.values())))

    def encode(texts):
        return np.array([table.get(t, [0.0] * dim) for t in texts], dtype=float)

    return MagicMock(side_effect=encode)


def _resolver(catalog, config=None, field="countries"):
    return EntityResolver(field, lambda: list(catalog), {**EMBEDDED, **(config or {})})


class TestExactLayer:
    def test_exact_catalog_value_passes_through(self):
        encode = MagicMock()
        with patch(ENCODE, encode):
            assert _resolver(GENRES, field="genres").resolve(["Comedy"]) == ["Comedy"]
        encode.assert_not_called()

    def test_case_and_whitespace_are_ignored_and_the_catalog_spelling_returned(self):
        with patch(ENCODE, MagicMock()):
            assert _resolver(COUNTRIES).resolve(["  united   KINGDOM "]) == ["United Kingdom"]

    def test_duplicates_collapse_and_order_is_kept(self):
        with patch(ENCODE, MagicMock()):
            assert _resolver(GENRES).resolve(["Horror", "comedy", "HORROR"]) == ["Horror", "Comedy"]

    def test_non_strings_and_blanks_are_skipped(self):
        with patch(ENCODE, MagicMock()):
            assert _resolver(COUNTRIES).resolve(["France", 42, None, "", "  "]) == ["France"]

    def test_empty_input_never_loads_the_catalog(self):
        catalog = MagicMock(return_value=COUNTRIES)
        assert EntityResolver("countries", catalog, EMBEDDED).resolve([]) == []
        catalog.assert_not_called()

    def test_catalog_values_load_once(self):
        catalog = MagicMock(return_value=COUNTRIES)
        resolver = EntityResolver("countries", catalog, EMBEDDED)
        resolver.resolve(["France"])
        resolver.resolve(["South Korea"])
        catalog.assert_called_once()


class TestSynonymLayer:
    def test_synonym_maps_to_the_catalog_value(self):
        config = {"synonyms": {"USA": "United States of America", "Korea": "South Korea"}}
        encode = MagicMock()
        with patch(ENCODE, encode):
            resolver = _resolver(COUNTRIES, config)
            assert resolver.resolve(["usa", " KOREA "]) == ["United States of America", "South Korea"]
        encode.assert_not_called()

    def test_synonym_target_is_matched_case_insensitively(self):
        with patch(ENCODE, MagicMock()):
            resolver = _resolver(GENRES, {"synonyms": {"sci-fi": "science fiction"}}, field="genres")
            assert resolver.resolve(["Sci-Fi"]) == ["Science Fiction"]

    def test_missing_target_warns_once_and_is_ignored(self, caplog):
        config = {"synonyms": {"Holland": "Netherlands"}, "match_threshold": 0.99}
        encode = _encoder({"Holland": [1.0, 0.0], **{c: [0.0, 1.0] for c in COUNTRIES}})
        with patch(ENCODE, encode), caplog.at_level(logging.INFO, logger="core.entity_resolution"):
            resolver = _resolver(COUNTRIES, config)
            assert resolver.resolve(["Holland"]) == []
            resolver.resolve(["Holland"])
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "Holland" in warnings[0].getMessage()
        assert "Netherlands" in warnings[0].getMessage()


class TestEmbeddingLayer:
    TABLE = {
        "United States of America": [1.0, 0.0, 0.0],
        "United Kingdom": [0.0, 1.0, 0.0],
        "South Korea": [0.0, 0.0, 1.0],
        "North Korea": [0.0, 0.1, 1.0],
        "France": [-1.0, 0.0, 0.0],
        "the states": [0.9, 0.1, 0.0],  # clearly USA
        "korean peninsula": [0.0, 0.02, 1.0],  # SK vs NK near-tie
        "atlantis": [0.0, -1.0, -1.0],  # nothing close
    }

    def test_confident_nearest_value_is_accepted(self):
        with patch(ENCODE, _encoder(self.TABLE)):
            assert _resolver(COUNTRIES).resolve(["the states"]) == ["United States of America"]

    def test_below_threshold_is_dropped_and_logged(self, caplog):
        with patch(ENCODE, _encoder(self.TABLE)), caplog.at_level(logging.INFO, logger="core.entity_resolution"):
            assert _resolver(COUNTRIES).resolve(["atlantis"]) == []
        message = caplog.records[-1].getMessage()
        assert caplog.records[-1].levelno == logging.INFO
        assert "countries" in message and "atlantis" in message

    def test_drop_log_names_the_intent_field_and_the_catalog(self, caplog):
        with patch(ENCODE, _encoder(self.TABLE)), caplog.at_level(logging.INFO, logger="core.entity_resolution"):
            _resolver(COUNTRIES).resolve(["atlantis"], "country_exclusions")
        message = caplog.records[-1].getMessage()
        assert "country_exclusions" in message and "countries" in message

    def test_near_tie_is_dropped_by_the_margin(self, caplog):
        with patch(ENCODE, _encoder(self.TABLE)), caplog.at_level(logging.INFO, logger="core.entity_resolution"):
            assert _resolver(COUNTRIES).resolve(["korean peninsula"]) == []
        message = caplog.records[-1].getMessage()
        assert "South Korea" in message and "North Korea" in message

    def test_zero_margin_accepts_the_near_tie(self):
        with patch(ENCODE, _encoder(self.TABLE)):
            assert _resolver(COUNTRIES, {"match_margin": 0.0}).resolve(["korean peninsula"]) == ["South Korea"]

    def test_catalog_embeddings_are_encoded_once(self):
        encode = _encoder(self.TABLE)
        with patch(ENCODE, encode):
            resolver = _resolver(COUNTRIES)
            resolver.resolve(["the states"])
            resolver.resolve(["atlantis", "korean peninsula"])
        catalog_calls = [c for c in encode.call_args_list if c.args[0] == COUNTRIES]
        assert len(catalog_calls) == 1

    def test_resolve_each_reports_layer_and_top_two(self):
        with patch(ENCODE, _encoder(self.TABLE)):
            resolver = _resolver(COUNTRIES, {"synonyms": {"USA": "United States of America"}})
            exact, synonym, embedded, dropped = resolver.resolve_each(
                ["France", "USA", "the states", "atlantis"]
            )
        assert (exact.value, exact.via) == ("France", "exact")
        assert (synonym.value, synonym.via) == ("United States of America", "synonym")
        assert (embedded.value, embedded.via) == ("United States of America", "embedding")
        assert embedded.top[0][0] == "United States of America"
        assert len(embedded.top) == 2
        assert (dropped.value, dropped.via) == (None, "dropped")

    def test_single_value_catalog_has_no_margin_to_check(self):
        with patch(ENCODE, _encoder(self.TABLE)):
            assert _resolver(["United States of America"]).resolve(["the states"]) == [
                "United States of America"
            ]


class TestWithoutEmbeddingFallback:
    LANGUAGES = ["en", "ko", "ja", "fr"]
    CONFIG = {"synonyms": {"korean": "ko", "japanese": "ja"}}

    def _resolver(self):
        return EntityResolver("languages", lambda: list(self.LANGUAGES), self.CONFIG)

    def test_exact_and_synonyms_resolve(self):
        with patch(ENCODE, MagicMock()):
            assert self._resolver().resolve(["KO", "japanese"]) == ["ko", "ja"]

    def test_unknown_is_dropped_without_encoding(self, caplog):
        encode = MagicMock()
        with patch(ENCODE, encode), caplog.at_level(logging.INFO, logger="core.entity_resolution"):
            assert self._resolver().resolve(["mandarin"]) == []
        encode.assert_not_called()
        assert "mandarin" in caplog.records[-1].getMessage()


class TestEmptyCatalog:
    def test_everything_is_dropped_without_encoding(self):
        encode = MagicMock()
        with patch(ENCODE, encode):
            resolver = _resolver([], {"synonyms": {"USA": "United States of America"}})
            assert resolver.resolve(["USA", "France"]) == []
        encode.assert_not_called()

    def test_empty_catalog_is_retried_on_the_next_call(self):
        """An empty catalog is a DB that has not been imported yet, not a
        fact to cache for the life of the process."""
        catalog = MagicMock(side_effect=[[], COUNTRIES])
        resolver = EntityResolver("countries", catalog, EMBEDDED)
        with patch(ENCODE, MagicMock()):
            assert resolver.resolve(["France"]) == []
            assert resolver.resolve(["France"]) == ["France"]


class TestNearest:
    def test_returns_the_top_two_with_scores(self):
        with patch(ENCODE, _encoder(TestEmbeddingLayer.TABLE)):
            [top] = _resolver(COUNTRIES).nearest(["the states"])
        assert [name for name, _ in top] == ["United States of America", "United Kingdom"]
        assert top[0][1] > top[1][1]

    def test_without_embedding_fallback_raises(self):
        resolver = EntityResolver("languages", lambda: ["en"], {"synonyms": {}})
        with pytest.raises(ValueError, match="languages"):
            resolver.nearest(["english"])
