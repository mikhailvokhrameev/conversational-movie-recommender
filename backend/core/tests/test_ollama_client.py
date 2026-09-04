import json
from unittest.mock import AsyncMock, patch, MagicMock

import httpx
import pytest

from core.ollama_client import (
    _ThinkStreamFilter,
    _parse_json_response,
    _strip_think,
    aclassify_message,
    aparse_intent,
    _extract_intent,
    _fallback_intent,
    parse_intent,
)


class TestClassifyMessage:
    @pytest.mark.asyncio
    async def test_returns_valid_category(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": json.dumps({"category": "follow_up"})}
        }

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("core.ollama_client.httpx.AsyncClient", return_value=mock_client):
            result = await aclassify_message("расскажи про первый фильм")
            assert result == "follow_up"

    @pytest.mark.asyncio
    async def test_invalid_category_defaults_to_new_search(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": json.dumps({"category": "invalid_type"})}
        }

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("core.ollama_client.httpx.AsyncClient", return_value=mock_client):
            result = await aclassify_message("test")
            assert result == "new_search"

    @pytest.mark.asyncio
    async def test_http_error_defaults_to_new_search(self):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("connection failed"))

        with patch("core.ollama_client.httpx.AsyncClient", return_value=mock_client):
            result = await aclassify_message("test")
            assert result == "new_search"


class TestExtractIntent:
    def test_extracts_all_fields(self):
        parsed = {
            "genres": ["Комедии"],
            "mood": "happy",
            "themes": ["семья"],
            "negations": ["Ужасы"],
            "reference_films": ["Один дома"],
            "country_exclusions": ["США"],
            "max_age_rating": 12,
            "min_release_year": 2015,
        }
        result = _extract_intent(parsed)
        assert result["mood"] == "happy"
        assert result["themes"] == ["семья"]
        assert result["reference_films"] == ["Один дома"]
        assert result["country_exclusions"] == ["США"]
        assert result["max_age_rating"] == 12.0
        assert result["min_release_year"] == 2015

    def test_missing_fields_default_to_empty(self):
        result = _extract_intent({})
        assert result["genres"] == []
        assert result["mood"] == ""
        assert result["themes"] == []
        assert result["negations"] == []
        assert result["reference_films"] == []
        assert result["country_exclusions"] == []
        assert result["max_age_rating"] is None
        assert result["min_release_year"] is None

    def test_invalid_hard_constraint_types_are_dropped(self):
        parsed = {
            "country_exclusions": ["США", 42, None, ""],
            "max_age_rating": "not-a-number",
            "min_release_year": "also-not-a-number",
        }
        result = _extract_intent(parsed)
        assert result["country_exclusions"] == ["США"]
        assert result["max_age_rating"] is None
        assert result["min_release_year"] is None


class TestFallbackIntent:
    def test_returns_empty_structure(self):
        result = _fallback_intent("any query")
        assert result["genres"] == []
        assert result["mood"] == ""
        assert result["themes"] == []
        assert result["negations"] == []
        assert result["reference_films"] == []
        assert result["country_exclusions"] == []
        assert result["max_age_rating"] is None
        assert result["min_release_year"] is None


class TestStripThink:
    """Hybrid reasoning models prefix answers with a <think> span. JSON-mode
    responses must survive one arriving despite the think=false flag."""

    def test_clean_json_untouched(self):
        assert _strip_think('{"a": 1}') == '{"a": 1}'

    def test_removes_leading_think_span(self):
        assert _strip_think('<think>hmm</think>{"a": 1}') == '{"a": 1}'

    def test_removes_multiple_spans(self):
        assert _strip_think('<think>x</think>{"a": 1}<think>y</think>') == '{"a": 1}'

    def test_unterminated_span_drops_remainder(self):
        """Ran out of tokens mid-reasoning: there is no answer to salvage."""
        assert _strip_think('<think>reasoning{"a": 1}') == ""

    def test_parse_json_response_tolerates_preamble(self):
        assert _parse_json_response('<think>hmm</think>\n{"category": "follow_up"}') == {
            "category": "follow_up"
        }


class TestThinkStreamFilter:
    """Reasoning tokens must never reach the user-visible SSE stream, even
    when a tag is split across chunk boundaries."""

    def _run(self, chunks):
        f = _ThinkStreamFilter()
        return "".join(f.feed(c) for c in chunks) + f.flush()

    def test_passes_through_plain_text(self):
        assert self._run(["Привет ", "мир"]) == "Привет мир"

    def test_removes_span_within_one_chunk(self):
        assert self._run(["<think>ага</think>Ответ"]) == "Ответ"

    def test_removes_span_split_across_chunks(self):
        assert self._run(["<thi", "nk>rea", "soning</thi", "nk>Ответ"]) == "Ответ"

    def test_removes_span_arriving_one_character_at_a_time(self):
        assert self._run(list("<think>hmm</think>OK")) == "OK"

    def test_keeps_text_on_both_sides_of_a_span(self):
        assert self._run(["До <think>x</think> после"]) == "До  после"

    def test_unterminated_span_emits_nothing(self):
        assert self._run(["<think>still reasoning"]) == ""

    def test_lone_angle_bracket_is_not_a_tag(self):
        assert self._run(["a < b"]) == "a < b"

    def test_partial_tag_that_is_not_a_tag_is_released(self):
        """'<th' is held back until 'anks' proves it was never a tag."""
        assert self._run(["<th", "anks"]) == "<thanks"

    def test_handles_two_spans(self):
        assert self._run(["<think>a</think>X<think>b</think>Y"]) == "XY"

    def test_empty_stream(self):
        assert self._run([]) == ""
