import json
from unittest.mock import AsyncMock, patch, MagicMock

import httpx
import numpy as np
import pytest

from core.ollama_client import (
    ExplanationSplitter,
    _ThinkStreamFilter,
    _parse_json_response,
    _strip_think,
    _normalize_genres,
    _build_chat_messages,
    _chat_sync,
    _classify_and_parse_payload,
    _explanation_payload,
    _intent_payload,
    _stream_chat_payload,
    aclassify_and_parse,
    astream_conversational,
    astream_explanation,
    astream_explanation_parts,
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


class _FakeStreamResponse:
    def __init__(self, lines):
        self._lines = lines
        self.raise_for_status = MagicMock()

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeStreamContextManager:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *args):
        return False


def _mock_streaming_client(lines):
    """Patch httpx.AsyncClient so client.stream(...) yields the given raw
    NDJSON lines (each already a JSON string, matching Ollama's streaming
    wire format)."""
    response = _FakeStreamResponse(lines)
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.stream = MagicMock(return_value=_FakeStreamContextManager(response))
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
            "min_vote_average": None,
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


    @pytest.mark.asyncio
    async def test_history_is_included_in_request_messages(self):
        content = json.dumps({"category": "follow_up", "semantic_query": "давай"})
        mock_client = _mock_ollama_responses(content)
        history = [
            {"role": "user", "content": "порекомендуй триллер"},
            {"role": "assistant", "content": "Вот Прибытие", "movies": ["Прибытие"]},
        ]

        with patch("core.ollama_client.httpx.AsyncClient", return_value=mock_client), \
             patch("core.ollama_client._normalize_genres", side_effect=lambda g: g):
            await aclassify_and_parse("давай", history)

        sent_messages = mock_client.post.call_args.kwargs["json"]["messages"]
        assert sent_messages[0]["role"] == "system"
        assert sent_messages[1] == {"role": "user", "content": "порекомендуй триллер"}
        assert sent_messages[2]["content"] == "Вот Прибытие\n\n(Recommended: Прибытие)"
        assert sent_messages[-1] == {"role": "user", "content": "давай"}

    @pytest.mark.asyncio
    async def test_no_history_argument_defaults_to_empty(self):
        content = json.dumps({"category": "new_search", "semantic_query": "q"})
        mock_client = _mock_ollama_responses(content)

        with patch("core.ollama_client.httpx.AsyncClient", return_value=mock_client), \
             patch("core.ollama_client._normalize_genres", side_effect=lambda g: g):
            result = await aclassify_and_parse("q")

        sent_messages = mock_client.post.call_args.kwargs["json"]["messages"]
        assert sent_messages[0]["role"] == "system"
        assert sent_messages[1] == {"role": "user", "content": "q"}
        assert len(sent_messages) == 2
        assert result.category == "new_search"


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
            "min_vote_average": "great",
            "min_release_year": "soon",
        })
        assert intent.genres == []
        assert intent.country_exclusions == ["США"]
        assert intent.min_vote_average is None
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
            "min_vote_average": 7.0,
            "min_release_year": 2015,
        }
        result = _extract_intent(parsed)
        assert result["mood"] == "happy"
        assert result["themes"] == ["семья"]
        assert result["reference_films"] == ["Один дома"]
        assert result["country_exclusions"] == ["США"]
        assert result["min_vote_average"] == 7.0
        assert result["min_release_year"] == 2015

    def test_missing_fields_default_to_empty(self):
        result = _extract_intent({})
        assert result["genres"] == []
        assert result["mood"] == ""
        assert result["themes"] == []
        assert result["negations"] == []
        assert result["reference_films"] == []
        assert result["country_exclusions"] == []
        assert result["min_vote_average"] is None
        assert result["min_release_year"] is None

    def test_invalid_hard_constraint_types_are_dropped(self):
        parsed = {
            "country_exclusions": ["США", 42, None, ""],
            "min_vote_average": "not-a-number",
            "min_release_year": "also-not-a-number",
        }
        result = _extract_intent(parsed)
        assert result["country_exclusions"] == ["США"]
        assert result["min_vote_average"] is None
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
        assert result["min_vote_average"] is None
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


class TestExplanationSplitter:
    """Each [[n]] marker moves the stream to the next movie's slot. Markers are
    split across chunk boundaries as readily as <think> tags, and a model that
    ignores the format must still get all of its text through."""

    def _run(self, chunks, count=2):
        splitter = ExplanationSplitter(count)
        parts = []
        for chunk in chunks:
            parts.extend(splitter.feed(chunk))
        parts.extend(splitter.flush())
        return parts

    def test_splits_intro_and_descriptions_in_one_chunk(self):
        assert self._run(["Вот подборка.\n[[1]]\nПро первый.\n[[2]]\nПро второй."]) == [
            (None, "Вот подборка.\n"),
            (0, "Про первый.\n"),
            (1, "Про второй."),
        ]

    def test_marker_split_across_three_chunks(self):
        """'[', '[1' and ']]' are only a marker once the last piece arrives."""
        assert self._run(["Интро [", "[1", "]]\nОписание"]) == [
            (None, "Интро "),
            (0, "Описание"),
        ]

    def test_no_markers_keeps_everything_in_the_intro(self):
        """Degradation path: the model ignored the format, so nothing is dropped."""
        assert self._run(["Один сплошной ", "ответ без маркеров."]) == [
            (None, "Один сплошной "),
            (None, "ответ без маркеров."),
        ]

    def test_out_of_range_marker_leaves_the_slot_alone(self):
        assert self._run(["[[1]]\nПервый. [[9]] Всё ещё первый."]) == [
            (0, "Первый. "),
            (0, "Всё ещё первый."),
        ]

    def test_text_before_the_first_marker_is_intro(self):
        assert self._run(["Нашёл кое-что интересное. [[1]] Первый."]) == [
            (None, "Нашёл кое-что интересное. "),
            (0, "Первый."),
        ]

    def test_flush_releases_a_held_back_tail(self):
        """'[[' never became a marker, so it was ordinary text."""
        splitter = ExplanationSplitter(2)
        assert splitter.feed("Текст [[") == [(None, "Текст ")]
        assert splitter.flush() == [(None, "[[")]

    def test_repeated_marker_does_not_rewind(self):
        assert self._run(["[[1]] Первый. [[1]] Всё ещё первый."]) == [
            (0, "Первый. "),
            (0, "Всё ещё первый."),
        ]

    def test_newline_after_a_marker_is_swallowed_across_a_boundary(self):
        """The boundary falls between the marker and its newline -- a description
        must never start with a blank line."""
        assert self._run(["Интро\n[[1]]", "\n", "Описание"]) == [
            (None, "Интро\n"),
            (0, "Описание"),
        ]


class TestBuildChatMessages:
    def test_no_history_returns_system_and_current_only(self):
        messages = _build_chat_messages(
            history=[], system_prompt="SYS", current_content="hi",
            max_turns=10, max_chars=1000,
        )
        assert messages == [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hi"},
        ]

    def test_includes_history_turns_in_order(self):
        history = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
        ]
        messages = _build_chat_messages(
            history=history, system_prompt="SYS", current_content="second",
            max_turns=10, max_chars=1000,
        )
        assert messages == [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]

    def test_caps_to_last_max_turns_pairs(self):
        history = [
            {"role": "user", "content": "turn1-user"},
            {"role": "assistant", "content": "turn1-assistant"},
            {"role": "user", "content": "turn2-user"},
            {"role": "assistant", "content": "turn2-assistant"},
            {"role": "user", "content": "turn3-user"},
            {"role": "assistant", "content": "turn3-assistant"},
        ]
        messages = _build_chat_messages(
            history=history, system_prompt="SYS", current_content="now",
            max_turns=2, max_chars=1000,
        )
        contents = [m["content"] for m in messages]
        assert contents == [
            "SYS", "turn2-user", "turn2-assistant",
            "turn3-user", "turn3-assistant", "now",
        ]

    def test_max_turns_zero_means_no_history_not_unlimited(self):
        history = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ]
        messages = _build_chat_messages(
            history=history, system_prompt="SYS", current_content="now",
            max_turns=0, max_chars=1000,
        )
        assert [m["content"] for m in messages] == ["SYS", "now"]

    def test_truncates_entry_content_exceeding_max_chars(self):
        history = [{"role": "user", "content": "x" * 50}]
        messages = _build_chat_messages(
            history=history, system_prompt="SYS", current_content="now",
            max_turns=10, max_chars=10,
        )
        assert messages[1]["content"] == "x" * 10

    def test_assistant_entry_with_movies_appended_to_content(self):
        history = [{
            "role": "assistant", "content": "Here you go",
            "movies": ["Film A", "Film B"],
        }]
        messages = _build_chat_messages(
            history=history, system_prompt="SYS", current_content="now",
            max_turns=10, max_chars=1000,
        )
        assert messages[1]["content"] == "Here you go\n\n(Recommended: Film A, Film B)"

    def test_skips_entries_with_missing_or_invalid_role(self):
        history = [
            {"content": "no role field"},
            {"role": "system", "content": "should not appear"},
            {"role": "user", "content": "valid"},
        ]
        messages = _build_chat_messages(
            history=history, system_prompt="SYS", current_content="now",
            max_turns=10, max_chars=1000,
        )
        assert messages == [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "valid"},
            {"role": "user", "content": "now"},
        ]


class TestStreamConversationalHistory:
    @pytest.mark.asyncio
    async def test_history_included_in_streamed_request(self):
        lines = [json.dumps({"message": {"content": "Sure thing"}})]
        mock_client = _mock_streaming_client(lines)
        history = [
            {"role": "user", "content": "порекомендуй триллер"},
            {"role": "assistant", "content": "Вот Прибытие", "movies": ["Прибытие"]},
        ]

        with patch("core.ollama_client.httpx.AsyncClient", return_value=mock_client):
            tokens = [t async for t in astream_conversational("давай", history)]

        assert "".join(tokens) == "Sure thing"
        sent_messages = mock_client.stream.call_args.kwargs["json"]["messages"]
        assert sent_messages[0]["role"] == "system"
        assert sent_messages[1] == {"role": "user", "content": "порекомендуй триллер"}
        assert sent_messages[2]["content"] == "Вот Прибытие\n\n(Recommended: Прибытие)"
        assert sent_messages[-1] == {"role": "user", "content": "давай"}

    @pytest.mark.asyncio
    async def test_no_history_argument_defaults_to_empty(self):
        lines = [json.dumps({"message": {"content": "Hi!"}})]
        mock_client = _mock_streaming_client(lines)

        with patch("core.ollama_client.httpx.AsyncClient", return_value=mock_client):
            tokens = [t async for t in astream_conversational("hi")]

        assert "".join(tokens) == "Hi!"
        sent_messages = mock_client.stream.call_args.kwargs["json"]["messages"]
        assert sent_messages == [
            {"role": "system", "content": sent_messages[0]["content"]},
            {"role": "user", "content": "hi"},
        ]


class TestStreamExplanationHistory:
    @pytest.mark.asyncio
    async def test_history_included_in_streamed_request(self):
        lines = [json.dumps({"message": {"content": "Great pick"}})]
        mock_client = _mock_streaming_client(lines)
        history = [
            {"role": "user", "content": "хочу триллер"},
            {"role": "assistant", "content": "Вот несколько триллеров", "movies": ["Прибытие"]},
        ]
        movies = [{"serial_name": "Семь", "genres": ["Thriller"], "description": "A detective story"}]

        with patch("core.ollama_client.httpx.AsyncClient", return_value=mock_client):
            tokens = [t async for t in astream_explanation("хочу еще триллер", movies, history)]

        assert "".join(tokens) == "Great pick"
        sent_messages = mock_client.stream.call_args.kwargs["json"]["messages"]
        assert sent_messages[0]["role"] == "system"
        assert sent_messages[1] == {"role": "user", "content": "хочу триллер"}
        assert sent_messages[2]["content"] == "Вот несколько триллеров\n\n(Recommended: Прибытие)"
        assert 'The user asked: "хочу еще триллер"' in sent_messages[-1]["content"]
        assert "Семь" in sent_messages[-1]["content"]

    @pytest.mark.asyncio
    async def test_no_history_argument_defaults_to_empty(self):
        lines = [json.dumps({"message": {"content": "OK"}})]
        mock_client = _mock_streaming_client(lines)
        movies = [{"serial_name": "Test Film", "genres": [], "description": ""}]

        with patch("core.ollama_client.httpx.AsyncClient", return_value=mock_client):
            tokens = [t async for t in astream_explanation("q", movies)]

        assert "".join(tokens) == "OK"
        sent_messages = mock_client.stream.call_args.kwargs["json"]["messages"]
        assert sent_messages[0]["role"] == "system"
        assert sent_messages[1]["role"] == "user"
        assert len(sent_messages) == 2

    @pytest.mark.asyncio
    async def test_system_message_carries_two_part_marker_format(self):
        lines = [json.dumps({"message": {"content": "x"}})]
        mock_client = _mock_streaming_client(lines)
        movies = [{"serial_name": "Test Film", "genres": [], "description": ""}]

        with patch("core.ollama_client.httpx.AsyncClient", return_value=mock_client):
            _ = [t async for t in astream_explanation("q", movies)]

        sent_messages = mock_client.stream.call_args.kwargs["json"]["messages"]
        assert "PART 1" in sent_messages[0]["content"]
        assert "PART 2" in sent_messages[0]["content"]
        assert "[[" in sent_messages[0]["content"]
        assert "[[1]]" in sent_messages[-1]["content"]

    @pytest.mark.asyncio
    async def test_parts_stream_yields_slots_with_history_in_request(self):
        lines = [
            json.dumps({"message": {"content": "Here are some thrillers.\n"}}),
            json.dumps({"message": {"content": "[[1]]\n"}}),
            json.dumps({"message": {"content": "A tense detective story."}}),
        ]
        mock_client = _mock_streaming_client(lines)
        history = [
            {"role": "user", "content": "хочу триллер"},
            {"role": "assistant", "content": "Вот триллеры", "movies": ["Прибытие"]},
        ]
        movies = [{"serial_name": "Семь", "genres": ["Thriller"], "description": "A detective story"}]

        with patch("core.ollama_client.httpx.AsyncClient", return_value=mock_client):
            parts = [p async for p in astream_explanation_parts("q", movies, history)]

        sent_messages = mock_client.stream.call_args.kwargs["json"]["messages"]
        assert sent_messages[1] == {"role": "user", "content": "хочу триллер"}
        assert parts
        assert all(isinstance(slot, (int, type(None))) and isinstance(text, str) for slot, text in parts)
        assert "Here are some thrillers." in "".join(t for s, t in parts if s is None)
        assert "A tense detective story." in "".join(t for s, t in parts if s == 0)


class TestHistoryLimitValidation:
    """params.yaml history limits < 1 must fail at settings load, never mean
    "unlimited" (the window is what bounds the prompt to llm.num_ctx)."""

    @pytest.mark.parametrize("key", ["history_max_turns", "history_max_chars_per_entry"])
    @pytest.mark.parametrize("bad", [0, -1, 1.5, "8", None, True])
    def test_rejects_values_below_one_or_non_int(self, key, bad):
        from recommender.settings import _validate_history_limits

        session = {"history_max_turns": 8, "history_max_chars_per_entry": 1000}
        session[key] = bad
        with pytest.raises(ValueError, match=key):
            _validate_history_limits(session)

    def test_accepts_minimum_of_one(self):
        from recommender.settings import _validate_history_limits

        _validate_history_limits({"history_max_turns": 1, "history_max_chars_per_entry": 1})


class TestOllamaOptions:
    """Every LLM call must carry the same options.num_ctx -- a differing value
    makes Ollama reload the model."""

    def test_intent_payload_has_num_ctx(self, settings):
        assert _intent_payload("a thriller")["options"] == {"num_ctx": settings.OLLAMA_NUM_CTX}

    def test_classify_and_parse_payload_has_num_ctx(self, settings):
        payload = _classify_and_parse_payload("hi", [])
        assert payload["options"] == {"num_ctx": settings.OLLAMA_NUM_CTX}

    def test_explanation_payload_has_num_ctx(self, settings):
        for stream in (False, True):
            payload = _explanation_payload("q", [], stream=stream)
            assert payload["options"] == {"num_ctx": settings.OLLAMA_NUM_CTX}

    def test_stream_chat_payload_has_num_ctx(self, settings):
        payload = _stream_chat_payload([{"role": "user", "content": "hi"}])
        assert payload["options"] == {"num_ctx": settings.OLLAMA_NUM_CTX}

    def test_chat_sync_payload_has_num_ctx(self, settings):
        response = MagicMock()
        response.json.return_value = {"message": {"content": "ok"}}
        with patch("core.ollama_client.httpx.post", return_value=response) as post:
            assert _chat_sync([{"role": "user", "content": "hi"}]) == "ok"
            _chat_sync([{"role": "user", "content": "hi"}], json_mode=True)
        for call in post.call_args_list:
            assert call.kwargs["json"]["options"] == {"num_ctx": settings.OLLAMA_NUM_CTX}
