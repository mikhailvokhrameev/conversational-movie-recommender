import os

import pytest

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "recommender.settings")


@pytest.fixture(autouse=True)
def _no_model_digest_lookup(monkeypatch):
    """The once-per-process Ollama digest lookup would otherwise make a real
    request on the first mocked LLM call of a test run; tests of the lookup
    itself reset this flag."""
    monkeypatch.setattr("core.ollama_client._digest_lookup_attempted", True)
