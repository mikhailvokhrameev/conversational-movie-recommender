"""Tests for the entity_resolution_report management command.

No DB and no model: the command's resolvers are replaced with real
EntityResolvers over stub catalogs, and encode_texts with a fixed vector
table."""

import json
from io import StringIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from core.entity_resolution import EntityResolver
from movies.management.commands.entity_resolution_report import default_probes_path

MODULE = "movies.management.commands.entity_resolution_report"

TABLE = {
    "Science Fiction": [1.0, 0.0, 0.0],
    "Horror": [0.0, 1.0, 0.0],
    "Comedy": [0.0, 0.0, 1.0],
    "scary": [0.1, 0.9, 0.0],
    "cartoon": [0.5, 0.0, 0.52],
}
EMBEDDED = {"match_threshold": 0.5, "match_margin": 0.05}


def _encode(texts):
    return np.array([TABLE.get(t, [0.0, 0.0, 0.0]) for t in texts], dtype=float)


def _resolvers():
    return {
        "genres": EntityResolver(
            "genres", lambda: ["Science Fiction", "Horror", "Comedy"],
            {**EMBEDDED, "synonyms": {"sci-fi": "Science Fiction"}},
        ),
        "languages": EntityResolver("languages", lambda: ["en", "ko"], {"synonyms": {"korean": "ko"}}),
    }


def _run(tmp_path, probes):
    path = tmp_path / "probes.json"
    path.write_text(json.dumps(probes))
    out = StringIO()
    with patch(f"{MODULE}.RESOLVERS", _resolvers()), \
         patch("core.entity_resolution.encode_texts", MagicMock(side_effect=_encode)):
        call_command("entity_resolution_report", "--probes", str(path), stdout=out)
    return out.getvalue()


def _line(output, probe):
    return next(line for line in output.splitlines() if line.strip().startswith(probe))


class TestReport:
    def test_prints_each_field_with_its_thresholds(self, tmp_path):
        out = _run(tmp_path, {"genres": ["scary"], "languages": ["korean"]})
        assert "genres (match_threshold 0.5, match_margin 0.05)" in out
        assert "languages (no embedding fallback)" in out

    def test_synonym_probe_shows_the_layer_and_the_embedding_top_two(self, tmp_path):
        line = _line(_run(tmp_path, {"genres": ["sci-fi"]}), "sci-fi")
        assert "Science Fiction" in line and "[synonym]" in line
        assert "top-2:" in line  # what the embedding alone would pick, for calibration

    def test_embedding_probe_shows_value_scores_and_margin(self, tmp_path):
        line = _line(_run(tmp_path, {"genres": ["scary"]}), "scary")
        assert "-> Horror" in line and "[embedding]" in line
        assert "'Horror' 0.994" in line and "'Science Fiction' 0.110" in line
        assert "margin 0.883" in line

    def test_dropped_probe_shows_dropped(self, tmp_path):
        line = _line(_run(tmp_path, {"genres": ["cartoon"]}), "cartoon")
        assert "-> (dropped)" in line
        assert "'Comedy' 0.721" in line and "'Science Fiction' 0.693" in line

    def test_field_without_embeddings_has_no_scores(self, tmp_path):
        out = _run(tmp_path, {"languages": ["korean", "mandarin"]})
        assert "[synonym]" in _line(out, "korean")
        mandarin = _line(out, "mandarin")
        assert "(dropped)" in mandarin and "top-2" not in mandarin

    def test_note_keys_are_skipped(self, tmp_path):
        out = _run(tmp_path, {"_note": "x", "languages": ["ko"]})
        assert "_note" not in out

    def test_unknown_field_fails(self, tmp_path):
        with pytest.raises(CommandError, match="moods"):
            _run(tmp_path, {"moods": ["happy"]})

    def test_malformed_probe_list_fails(self, tmp_path):
        with pytest.raises(CommandError, match="genres"):
            _run(tmp_path, {"genres": "sci-fi"})

    def test_missing_file_fails(self, tmp_path):
        with pytest.raises(CommandError, match="nope.json"):
            call_command("entity_resolution_report", "--probes", str(tmp_path / "nope.json"))


class TestShippedProbes:
    def test_covers_every_resolver_with_the_planned_probes(self):
        from core.ollama_client import RESOLVERS

        probes = json.loads(default_probes_path().read_text())
        fields = {k: v for k, v in probes.items() if not k.startswith("_")}
        assert set(fields) == set(RESOLVERS)
        assert fields["countries"] == ["USA", "Britain", "Korea", "Soviet Union", "West Germany", "Holland"]
        assert fields["genres"] == ["sci-fi", "rom-com", "scary", "cartoon"]
        assert fields["languages"] == ["korean", "mandarin"]
