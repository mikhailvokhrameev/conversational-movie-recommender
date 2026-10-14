"""params.yaml values that settings.py refuses to load when they are wrong."""

import pytest

from recommender.settings import _validate_revision


class TestRevision:
    @pytest.mark.parametrize("ok", [None, "5617a9f61b028005a4858fdac845db406aefb181", "main"])
    def test_accepts_null_or_a_string(self, ok):
        _validate_revision("embedding", {"revision": ok})

    @pytest.mark.parametrize("bad", ["", "  ", 123, True, ["abc"]])
    def test_rejects_anything_else(self, bad):
        with pytest.raises(ValueError, match="reranking.revision"):
            _validate_revision("reranking", {"revision": bad})

    def test_shipped_params_pin_both_models(self):
        from django.conf import settings

        for revision in (settings.EMBEDDING_REVISION, settings.RERANKER_REVISION):
            assert isinstance(revision, str) and len(revision) == 40
