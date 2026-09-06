import json
from unittest.mock import AsyncMock, patch, MagicMock

import httpx
import numpy as np
import pytest

from core.ollama_client import (
    _ThinkStreamFilter,
    _parse_json_response,
    _strip_think,
    _normalize_genres,
    aclassify_and_parse,
    check_explanation_titles,
    CATALOG_GENRES,
    MessageIntent,
    _extract_intent,
    _fallback_intent,
    parse_intent,
)


def _mock_ollama_responses(*contents):
    """Patch httpx.AsyncClient so successive posts return the given response bodies."""
    responses = []
    for content in contents:
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"message": {"content": content}}
        responses.append(mock_response)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(side_effect=responses)
    return mock_client


class TestClassifyAndParse:
    @pytest.mark.asyncio
    async def test_returns_valid_intent_and_category(self):
        content = json.dumps({
            "category": "new_search",
            "semantic_query": "весёлая комедия",
            "genres": ["Комедии"],
            "mood": "happy",
            "themes": [],
            "negations": [],
            "reference_films": [],
            "country_exclusions": [],
            "max_age_rating": None,
            "min_release_year": None,
        })
        mock_client = _mock_ollama_responses(content)

        with patch("core.ollama_client.httpx.AsyncClient", return_value=mock_client), \
             patch("core.ollama_client._normalize_genres", side_effect=lambda g: g):
            result = await aclassify_and_parse("хочу весёлую комедию")

        assert isinstance(result, MessageIntent)
        assert result.category == "new_search"
        assert result.semantic_query == "весёлая комедия"
        assert result.genres == ["Комедии"]
        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_invalid_category_retries_once_then_succeeds(self):
        bad = json.dumps({"category": "not_a_real_category", "semantic_query": "x"})
        good = json.dumps({"category": "follow_up", "semantic_query": "x"})
        mock_client = _mock_ollama_responses(bad, good)

        with patch("core.ollama_client.httpx.AsyncClient", return_value=mock_client), \
             patch("core.ollama_client._normalize_genres", side_effect=lambda g: g):
            result = await aclassify_and_parse("расскажи про первый фильм")

        assert result.category == "follow_up"
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_invalid_category_repair_also_fails_uses_fallback(self):
        bad = json.dumps({"category": "not_a_real_category"})
        mock_client = _mock_ollama_responses(bad, bad)

        with patch("core.ollama_client.httpx.AsyncClient", return_value=mock_client), \
             patch("core.ollama_client._normalize_genres", side_effect=lambda g: g):
            result = await aclassify_and_parse("test message")

        assert result.category == "new_search"
        assert result.semantic_query == "test message"
        assert result.genres == []
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_http_error_uses_fallback_without_retry(self):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("connection failed"))

        with patch("core.ollama_client.httpx.AsyncClient", return_value=mock_client):
            result = await aclassify_and_parse("test")

        assert result.category == "new_search"
        assert result.semantic_query == "test"
        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_empty_semantic_query_falls_back_to_raw_message(self):
        content = json.dumps({"category": "new_search", "semantic_query": ""})
        mock_client = _mock_ollama_responses(content)

        with patch("core.ollama_client.httpx.AsyncClient", return_value=mock_client), \
             patch("core.ollama_client._normalize_genres", side_effect=lambda g: g):
            result = await aclassify_and_parse("хочу комедию")

        assert result.semantic_query == "хочу комедию"

    @pytest.mark.asyncio
    async def test_malformed_json_on_first_request_uses_fallback_without_retry(self):
        mock_client = _mock_ollama_responses("not valid json")

        with patch("core.ollama_client.httpx.AsyncClient", return_value=mock_client):
            result = await aclassify_and_parse("test")

        assert result.category == "new_search"
        assert result.semantic_query == "test"
        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_genres_and_negations_are_normalized(self):
        content = json.dumps({
            "category": "new_search",
            "semantic_query": "q",
            "genres": ["комедия"],
            "negations": ["хоррор"],
        })
        mock_client = _mock_ollama_responses(content)

        with patch("core.ollama_client.httpx.AsyncClient", return_value=mock_client), \
             patch("core.ollama_client._normalize_genres", side_effect=lambda g: [x.upper() for x in g]):
            result = await aclassify_and_parse("q")

        assert result.genres == ["КОМЕДИЯ"]
        assert result.negations == ["ХОРРОР"]


class TestCheckExplanationTitles:
    def test_no_suspicious_titles_when_only_top_titles_mentioned(self):
        explanation = "Рекомендую Фильм А, отличная драма."
        result = check_explanation_titles(
            explanation, top_titles=["Фильм А"], candidate_titles=["Фильм А", "Фильм Б"]
        )
        assert result == []

    def test_flags_candidate_title_outside_top_titles(self):
        explanation = "Кстати, Фильм Б тоже интересный, но рекомендую Фильм А."
        result = check_explanation_titles(
            explanation, top_titles=["Фильм А"], candidate_titles=["Фильм А", "Фильм Б"]
        )
        assert result == ["Фильм Б"]

    def test_case_insensitive_match(self):
        explanation = "рекомендую фильм б"
        result = check_explanation_titles(
            explanation, top_titles=["Фильм А"], candidate_titles=["Фильм А", "Фильм Б"]
        )
        assert result == ["Фильм Б"]

    def test_empty_explanation_returns_no_suspicious_titles(self):
        result = check_explanation_titles(
            "", top_titles=["Фильм А"], candidate_titles=["Фильм А", "Фильм Б"]
        )
        assert result == []


class TestMessageIntentValidation:
    def test_tolerates_malformed_llm_field_shapes(self):
        intent = MessageIntent.model_validate({
            "category": "new_search",
            "genres": "Комедии",
            "country_exclusions": ["США", 42, None, ""],
            "max_age_rating": "kids",
            "min_release_year": "soon",
        })
        assert intent.genres == []
        assert intent.country_exclusions == ["США"]
        assert intent.max_age_rating is None
        assert intent.min_release_year is None


class TestNormalizeGenres:
    def test_exact_match_passthrough(self):
        with patch(
            "core.ollama_client._get_genre_embeddings",
            return_value=[np.zeros(4)] * len(CATALOG_GENRES),
        ):
            assert _normalize_genres(["Комедии"]) == ["Комедии"]

    def test_drops_below_threshold(self, settings):
        settings.GENRE_MATCH_THRESHOLD = 0.99
        with patch(
            "core.ollama_client._get_genre_embeddings",
            return_value=[np.array([0.0, 1.0])] * len(CATALOG_GENRES),
        ), patch(
            "core.ollama_client.encode_texts",
            return_value=[np.array([1.0, 0.0])],
        ):
            assert _normalize_genres(["totally unrelated term"]) == []


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
