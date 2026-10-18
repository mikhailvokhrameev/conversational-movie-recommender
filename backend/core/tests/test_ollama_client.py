import datetime
import json
import logging
from unittest.mock import AsyncMock, patch, MagicMock

import httpx
import numpy as np
import pytest
from django.conf import settings
from pydantic import ValidationError

from core.ollama_client import (
    EXPLANATION_PROMPT,
    EXPLANATION_SYSTEM_PROMPT,
    ExplanationSplitter,
    _ThinkStreamFilter,
    _parse_json_response,
    _strip_think,
    _build_chat_messages,
    _chat_sync,
    _classify_and_parse_payload,
    _explanation_payload,
    _intent_payload,
    _log_call_stats,
    _log_model_digest_once,
    _alog_model_digest_once,
    _StreamStats,
    _build_movies_context,
    _ollama_options,
    _stream_chat_payload,
    aclassify_and_parse,
    astream_conversational,
    astream_explanation,
    astream_explanation_parts,
    check_explanation_titles,
    CATALOG_GENRES,
    CLARIFY_SYSTEM_PROMPT,
    CLASSIFY_AND_PARSE_SYSTEM_PROMPT,
    CONVERSATIONAL_SYSTEM_PROMPT,
    CLASSIFY_PARSE_SCHEMA,
    INTENT_SCHEMA,
    NON_FILTER_FIELDS,
    RESOLVED_FIELDS,
    RESOLVERS,
    MessageIntent,
    _extract_intent,
    _fallback_intent,
    _resolve_fields,
    parse_intent,
)
from core.entity_resolution import catalog_countries, catalog_languages


@pytest.fixture(autouse=True)
def _no_catalog_db():
    """The classify request reads the catalog's year bounds from the DB; no
    test here has one, and the bounds are None-safe (the rule is omitted)."""
    with patch("core.ollama_client.catalog_year_range", return_value=None):
        yield


def _unresolved(values):
    """_resolve_fields stand-in: every resolved field passes through as is."""
    return {field: list(values.get(field) or []) for field in RESOLVED_FIELDS}


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


class TestStandaloneQuery:
    def test_standalone_query_is_in_the_classify_schema_not_the_intent_schema(self):
        assert "standalone_query" in CLASSIFY_PARSE_SCHEMA["properties"]
        assert "standalone_query" in CLASSIFY_PARSE_SCHEMA["required"]
        assert "standalone_query" not in INTENT_SCHEMA["properties"]

    @pytest.mark.asyncio
    async def test_blank_standalone_query_falls_back_to_message(self):
        content = json.dumps({
            "category": "new_search",
            "semantic_query": "a comedy",
            "standalone_query": "  ",
        })
        mock_client = _mock_ollama_responses(content)

        with patch("core.ollama_client.httpx.AsyncClient", return_value=mock_client), \
             patch("core.ollama_client._resolve_fields", side_effect=_unresolved):
            result = await aclassify_and_parse("something funny")

        assert result.standalone_query == "something funny"

    @pytest.mark.asyncio
    async def test_fallback_intent_carries_the_message(self):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("connection failed"))

        with patch("core.ollama_client.httpx.AsyncClient", return_value=mock_client):
            result = await aclassify_and_parse("a sad drama")

        assert result.standalone_query == result.semantic_query == "a sad drama"

    def test_prompt_examples_include_standalone_query(self):
        prompt = CLASSIFY_AND_PARSE_SYSTEM_PROMPT.replace("{{", "{").replace("}}", "}")
        examples = [
            json.loads(line) for line in prompt.splitlines() if line.startswith('{"category"')
        ]
        assert len(examples) >= 5
        assert all("standalone_query" in ex for ex in examples)


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
             patch("core.ollama_client._resolve_fields", side_effect=_unresolved):
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
             patch("core.ollama_client._resolve_fields", side_effect=_unresolved):
            result = await aclassify_and_parse("расскажи про первый фильм")

        assert result.category == "follow_up"
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_invalid_category_repair_also_fails_uses_fallback(self):
        bad = json.dumps({"category": "not_a_real_category"})
        mock_client = _mock_ollama_responses(bad, bad)

        with patch("core.ollama_client.httpx.AsyncClient", return_value=mock_client), \
             patch("core.ollama_client._resolve_fields", side_effect=_unresolved):
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
             patch("core.ollama_client._resolve_fields", side_effect=_unresolved):
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
    async def test_every_resolved_field_goes_through_resolution(self):
        content = json.dumps({
            "category": "new_search",
            "semantic_query": "q",
            "genres": ["sci-fi"],
            "negations": ["scary"],
            "country_inclusions": ["Korea"],
            "country_exclusions": ["USA"],
            "original_languages": ["korean"],
            "reference_films": ["Oldboy"],
        })
        mock_client = _mock_ollama_responses(content)
        resolve = MagicMock(side_effect=lambda values: {
            field: [v.upper() for v in values[field]] for field in RESOLVED_FIELDS
        })

        with patch("core.ollama_client.httpx.AsyncClient", return_value=mock_client), \
             patch("core.ollama_client._resolve_fields", resolve):
            result = await aclassify_and_parse("q")

        resolve.assert_called_once()
        assert result.genres == ["SCI-FI"]
        assert result.negations == ["SCARY"]
        assert result.country_inclusions == ["KOREA"]
        assert result.country_exclusions == ["USA"]
        assert result.original_languages == ["KOREAN"]
        assert result.reference_films == ["Oldboy"]


    @pytest.mark.asyncio
    async def test_history_is_included_in_request_messages(self):
        content = json.dumps({"category": "follow_up", "semantic_query": "давай"})
        mock_client = _mock_ollama_responses(content)
        history = [
            {"role": "user", "content": "порекомендуй триллер"},
            {"role": "assistant", "content": "Вот Прибытие", "movies": ["Прибытие"]},
        ]

        with patch("core.ollama_client.httpx.AsyncClient", return_value=mock_client), \
             patch("core.ollama_client._resolve_fields", side_effect=_unresolved):
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
             patch("core.ollama_client._resolve_fields", side_effect=_unresolved):
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

    def test_accepts_clarify_category(self):
        intent = MessageIntent.model_validate({"category": "clarify", "semantic_query": "movies"})
        assert intent.category == "clarify"

    def test_rejects_unknown_category(self):
        with pytest.raises(ValidationError):
            MessageIntent.model_validate({"category": "chit_chat"})


class TestConversationalSystemPrompt:
    async def _system_prompt_sent(self, **kwargs):
        sent = {}

        def _capture(messages):
            sent["messages"] = messages
            raise httpx.ConnectError("no ollama in tests")

        with patch("core.ollama_client._stream_chat_payload", side_effect=_capture):
            tokens = [t async for t in astream_conversational("movies", [], **kwargs)]
        assert tokens == []
        return sent["messages"][0]["content"]

    @pytest.mark.asyncio
    async def test_defaults_to_conversational_prompt(self):
        assert await self._system_prompt_sent() == CONVERSATIONAL_SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_uses_given_system_prompt(self):
        prompt = await self._system_prompt_sent(system_prompt=CLARIFY_SYSTEM_PROMPT)
        assert prompt == CLARIFY_SYSTEM_PROMPT


def _prompt_examples():
    prompt = CLASSIFY_AND_PARSE_SYSTEM_PROMPT.replace("{{", "{").replace("}}", "}")
    return [json.loads(line) for line in prompt.splitlines() if line.startswith('{"category"')]


class TestResultCount:
    def test_is_in_the_classify_schema_not_the_intent_schema(self):
        assert "result_count" in CLASSIFY_PARSE_SCHEMA["required"]
        assert "result_count" in NON_FILTER_FIELDS
        assert "result_count" not in INTENT_SCHEMA["properties"]

    def test_defaults_to_none_and_coerces_junk(self):
        assert MessageIntent.model_validate({"category": "new_search"}).result_count is None
        assert MessageIntent.model_validate({"category": "new_search", "result_count": "3"}).result_count == 3
        assert MessageIntent.model_validate({"category": "new_search", "result_count": "many"}).result_count is None

    def test_every_prompt_example_has_the_field(self):
        assert all("result_count" in ex for ex in _prompt_examples())

    def test_prompt_documents_it_with_the_one_more_movie_example(self):
        assert '"result_count"' in CLASSIFY_AND_PARSE_SYSTEM_PROMPT
        example = next(ex for ex in _prompt_examples() if ex["result_count"] == 1)
        assert example["category"] == "refinement"


class TestUnsupportedConstraints:
    def test_is_in_the_classify_schema_not_the_intent_schema(self):
        assert "unsupported_constraints" in CLASSIFY_PARSE_SCHEMA["required"]
        assert "unsupported_constraints" not in INTENT_SCHEMA["properties"]

    def test_non_filter_fields_cover_every_field_outside_the_intent_schema(self):
        assert set(NON_FILTER_FIELDS) == set(CLASSIFY_PARSE_SCHEMA["properties"]) - set(INTENT_SCHEMA["properties"])

    def test_drops_blank_and_non_string_entries(self):
        intent = MessageIntent.model_validate({
            "category": "new_search", "unsupported_constraints": ["silent", "", 3, None],
        })
        assert intent.unsupported_constraints == ["silent"]

    def test_every_prompt_example_has_the_field(self):
        assert all("unsupported_constraints" in ex for ex in _prompt_examples())

    def test_actor_example_keeps_the_era_and_leaves_the_actor_out_of_standalone_query(self):
        example = next(
            ex for ex in _prompt_examples() if ex["unsupported_constraints"] == ["with Keanu Reeves"]
        )
        assert (example["min_release_year"], example["max_release_year"]) == (1950, 1959)
        assert "Keanu" not in example["standalone_query"]
        assert "Keanu" not in example["semantic_query"]

    def test_silent_example_flags_silent(self):
        example = next(ex for ex in _prompt_examples() if ex["unsupported_constraints"] == ["silent"])
        assert "silent" not in example["standalone_query"]

    def test_single_turn_silent_example_flags_silent_and_keeps_the_rest(self):
        example = next(
            ex for ex in _prompt_examples()
            if ex["unsupported_constraints"] == ["silent"] and "documentar" in ex["standalone_query"]
            and ex["category"] == "new_search"
        )
        assert "silent" not in example["standalone_query"]
        assert "silent" not in example["semantic_query"]
        assert example["max_runtime"] == 20
        assert example["genres"] == ["Animation", "Documentary"]


    def test_standalone_query_field_says_to_leave_unsupported_constraints_out(self):
        line = next(
            line for line in CLASSIFY_AND_PARSE_SYSTEM_PROMPT.splitlines()
            if line.startswith('- "standalone_query"')
        )
        assert "unsupported_constraints" in line


class TestClassifyPromptRules:
    def test_a_complaint_about_shown_movies_is_not_taught_as_a_refinement(self):
        assert not any(
            ex["category"] == "refinement" and ex["unsupported_constraints"] == ["silent"]
            for ex in _prompt_examples()
        )

    def test_complaint_examples_classify_as_general_chat(self):
        prompt = CLASSIFY_AND_PARSE_SYSTEM_PROMPT
        assert 'Latest message: "They are not silent"' in prompt
        assert "bro I told you" in prompt
        chat = [ex for ex in _prompt_examples() if ex["category"] == "general_chat"]
        assert len(chat) >= 2
        assert all(ex["result_count"] is None for ex in chat)

    def test_refinement_means_different_movies_not_a_report_that_the_last_ones_did_not_fit(self):
        line = next(
            line for line in CLASSIFY_AND_PARSE_SYSTEM_PROMPT.splitlines() if line.startswith('- "refinement"')
        )
        assert "different movies" in line
        assert "a new condition" in line
        assert "did not fit" in line

    def test_general_chat_covers_complaints_about_the_previous_answer(self):
        line = next(
            line for line in CLASSIFY_AND_PARSE_SYSTEM_PROMPT.splitlines() if line.startswith('- "general_chat"')
        )
        assert "why only one movie" in line
        assert "I asked for" in line

    def test_negations_hold_only_genres_rejected_by_name(self):
        line = next(
            line for line in CLASSIFY_AND_PARSE_SYSTEM_PROMPT.splitlines() if line.startswith('- "negations"')
        )
        assert "by name" in line
        assert "not about cars" in line

    def test_classic_without_an_era_sets_no_year(self):
        line = next(
            line for line in CLASSIFY_AND_PARSE_SYSTEM_PROMPT.splitlines()
            if line.startswith('- "max_release_year"')
        )
        assert '"classic"' in line and '"old-school"' in line
        assert "no year" in line


class TestClassifyPromptClarify:
    def test_lists_clarify_category_and_moves_what_should_i_watch(self):
        new_search_line = next(
            line for line in CLASSIFY_AND_PARSE_SYSTEM_PROMPT.splitlines() if line.startswith('- "new_search"')
        )
        assert "what should I watch" not in new_search_line
        assert '- "clarify"' in CLASSIFY_AND_PARSE_SYSTEM_PROMPT
        assert '"category": "clarify"' in CLASSIFY_AND_PARSE_SYSTEM_PROMPT


class TestResolveFields:
    def test_each_field_uses_the_resolver_for_its_catalog(self):
        assert RESOLVED_FIELDS == {
            "genres": "genres", "negations": "genres",
            "country_inclusions": "countries", "country_exclusions": "countries",
            "original_languages": "languages",
        }
        assert RESOLVERS["genres"]._catalog_values() == CATALOG_GENRES
        assert RESOLVERS["countries"]._catalog_values is catalog_countries
        assert RESOLVERS["languages"]._catalog_values is catalog_languages
        assert not RESOLVERS["languages"].uses_embeddings
        assert RESOLVERS["genres"].uses_embeddings and RESOLVERS["countries"].uses_embeddings

    def test_resolvers_read_their_params(self, settings):
        config = settings.INTENT_RESOLUTION["countries"]
        resolver = RESOLVERS["countries"]
        assert (resolver.match_threshold, resolver.match_margin) == (
            config["match_threshold"], config["match_margin"]
        )

    def test_resolves_each_field_through_its_resolver(self):
        fakes = {name: MagicMock() for name in ("genres", "countries", "languages")}
        for name, fake in fakes.items():
            fake.resolve.side_effect = lambda raw, field, name=name: [f"{name}:{v}" for v in raw]
        with patch.dict("core.ollama_client.RESOLVERS", fakes):
            result = _resolve_fields({
                "genres": ["a"], "negations": ["b"], "country_inclusions": ["c"],
                "country_exclusions": ["d"], "original_languages": ["e"], "mood": "x",
            })
        assert result == {
            "genres": ["genres:a"], "negations": ["genres:b"],
            "country_inclusions": ["countries:c"], "country_exclusions": ["countries:d"],
            "original_languages": ["languages:e"],
        }

    def test_missing_or_null_fields_resolve_to_empty(self):
        fakes = {name: MagicMock() for name in ("genres", "countries", "languages")}
        for fake in fakes.values():
            fake.resolve.side_effect = lambda raw, field: list(raw)
        with patch.dict("core.ollama_client.RESOLVERS", fakes):
            assert _resolve_fields({"genres": None}) == {field: [] for field in RESOLVED_FIELDS}


class TestExtractIntent:
    def test_extracts_all_fields(self):
        parsed = {
            "genres": ["Comedy"],
            "mood": "happy",
            "themes": ["family"],
            "negations": ["Horror"],
            "reference_films": ["Home Alone"],
            "country_exclusions": ["United States of America"],
            "min_vote_average": 7.0,
            "min_release_year": 2015,
        }
        with patch("core.ollama_client._resolve_fields", side_effect=_unresolved):
            result = _extract_intent(parsed)
        assert result["genres"] == ["Comedy"]
        assert result["negations"] == ["Horror"]
        assert result["mood"] == "happy"
        assert result["themes"] == ["family"]
        assert result["reference_films"] == ["Home Alone"]
        assert result["country_exclusions"] == ["United States of America"]
        assert result["min_vote_average"] == 7.0
        assert result["min_release_year"] == 2015

    def test_resolved_fields_come_from_resolution(self):
        resolved = {field: [f"resolved {field}"] for field in RESOLVED_FIELDS}
        with patch("core.ollama_client._resolve_fields", return_value=resolved) as resolve:
            result = _extract_intent({"genres": ["sci-fi"], "country_inclusions": ["Korea"]})
        resolve.assert_called_once_with({"genres": ["sci-fi"], "country_inclusions": ["Korea"]})
        for field in RESOLVED_FIELDS:
            assert result[field] == [f"resolved {field}"]

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
            "min_vote_average": "not-a-number",
            "min_release_year": "also-not-a-number",
        }
        result = _extract_intent(parsed)
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
    """Hybrid reasoning models prefix answers with a <think> span. Schema-constrained
    replies must survive one arriving despite the think=false flag."""

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

    def test_placeholder_in_the_intro_is_dropped(self):
        """The model echoed the prompt's shape literally: '<opening line>' is not text."""
        assert self._run(["<opening line>\n[[1]]\nПервый."]) == [(0, "Первый.")]

    def test_placeholder_in_a_description_is_dropped(self):
        assert self._run(["Интро.\n[[1]]\n<description of movie 1>\nПро первый."]) == [
            (None, "Интро.\n"),
            (0, "Про первый."),
        ]

    def test_placeholder_split_across_chunks_is_dropped(self):
        """'<descr' could still close into a placeholder, so it is held back."""
        assert self._run(["Интро <descr", "iption of movie 1> и ещё."]) == [
            (None, "Интро "),
            (None, "и ещё."),
        ]

    def test_angle_bracket_that_never_closes_is_released(self):
        """A '<' the line ends without closing was ordinary text."""
        assert self._run(["a < b", " and more\n"]) == [
            (None, "a "),
            (None, "< b and more\n"),
        ]

    def test_flush_releases_an_unclosed_angle_bracket(self):
        splitter = ExplanationSplitter(2)
        assert splitter.feed("Текст <") == [(None, "Текст ")]
        assert splitter.flush() == [(None, "<")]

    def test_explanation_prompts_carry_no_angle_bracket_placeholders(self):
        """The literal '<opening line>' shape is what the model used to copy back."""
        for prompt in (EXPLANATION_PROMPT, EXPLANATION_SYSTEM_PROMPT):
            assert "<" not in prompt and ">" not in prompt
            assert "[[1]]" in prompt and "[[2]]" in prompt


    def test_system_prompt_forbids_claiming_unshown_conditions(self):
        """The opening line may only restate what the listed facts show."""
        assert "never state that the movies meet a condition" in " ".join(EXPLANATION_SYSTEM_PROMPT.split())
        assert "listed facts" in EXPLANATION_SYSTEM_PROMPT


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

    def test_assistant_entry_with_empty_content_and_movies_has_no_leading_blank_lines(self):
        history = [{"role": "assistant", "content": "", "movies": ["Film A", "Film B"]}]
        messages = _build_chat_messages(
            history=history, system_prompt="SYS", current_content="now",
            max_turns=10, max_chars=1000,
        )
        assert messages[1]["content"] == "(Recommended: Film A, Film B)"

    @staticmethod
    def _facts_entry(content="Here you go", titles=("Film A", "Film B"), **fact_overrides):
        facts = [
            {"title": t, "year": 1999, "genres": ["Drama", "Crime"], "runtime": 142,
             "description": f"About {t}.", **fact_overrides}
            for t in titles
        ]
        return {"role": "assistant", "content": content, "movies": list(titles), "movie_facts": facts}

    def test_latest_entry_with_facts_renders_facts_block(self):
        messages = _build_chat_messages(
            history=[{"role": "user", "content": "q"}, self._facts_entry()],
            system_prompt="SYS", current_content="now", max_turns=10, max_chars=1000,
        )
        assert messages[2]["content"] == (
            "Here you go\n\n(Recommended movies:\n"
            "1. Film A (1999) — Drama, Crime — 142 min: About Film A.\n"
            "2. Film B (1999) — Drama, Crime — 142 min: About Film B.)"
        )

    def test_older_entry_with_facts_stays_titles_only(self):
        history = [
            {"role": "user", "content": "q1"},
            self._facts_entry(content="First set", titles=("Old A", "Old B")),
            {"role": "user", "content": "q2"},
            self._facts_entry(content="Second set", titles=("New A",)),
        ]
        messages = _build_chat_messages(
            history=history, system_prompt="SYS", current_content="now",
            max_turns=10, max_chars=1000,
        )
        assert messages[2]["content"] == "First set\n\n(Recommended: Old A, Old B)"
        assert "1. New A (1999)" in messages[4]["content"]

    def test_latest_facts_entry_is_chosen_among_later_entries_without_facts(self):
        history = [
            self._facts_entry(),
            {"role": "user", "content": "thanks"},
            {"role": "assistant", "content": "You're welcome"},
        ]
        messages = _build_chat_messages(
            history=history, system_prompt="SYS", current_content="now",
            max_turns=10, max_chars=1000,
        )
        assert "1. Film A (1999)" in messages[1]["content"]
        assert messages[3]["content"] == "You're welcome"

    def test_legacy_entry_without_facts_renders_exactly_as_before(self):
        history = [{"role": "assistant", "content": "Here you go", "movies": ["Film A", "Film B"]}]
        messages = _build_chat_messages(
            history=history, system_prompt="SYS", current_content="now",
            max_turns=10, max_chars=1000,
        )
        assert messages[1]["content"] == "Here you go\n\n(Recommended: Film A, Film B)"

    def test_legacy_entry_after_facts_entry_keeps_titles_only(self):
        # The facts block goes to the latest entry that HAS facts; a newer
        # legacy-shaped entry must not steal it nor gain one.
        history = [
            self._facts_entry(),
            {"role": "assistant", "content": "Newer", "movies": ["Film C"]},
        ]
        messages = _build_chat_messages(
            history=history, system_prompt="SYS", current_content="now",
            max_turns=10, max_chars=1000,
        )
        assert "Recommended movies" in messages[1]["content"]
        assert messages[2]["content"] == "Newer\n\n(Recommended: Film C)"

    def test_empty_content_with_facts_has_no_leading_blank_lines(self):
        messages = _build_chat_messages(
            history=[self._facts_entry(content="")], system_prompt="SYS",
            current_content="now", max_turns=10, max_chars=1000,
        )
        assert messages[1]["content"].startswith("(Recommended movies:\n1. Film A")

    def test_facts_block_is_not_cut_by_the_reply_char_cap(self):
        messages = _build_chat_messages(
            history=[self._facts_entry(content="x" * 50)], system_prompt="SYS",
            current_content="now", max_turns=10, max_chars=10,
        )
        content = messages[1]["content"]
        assert content.startswith("x" * 10 + "\n\n(Recommended movies:")
        assert "2. Film B (1999) — Drama, Crime — 142 min: About Film B.)" in content

    def test_missing_year_runtime_genres_and_description_are_omitted(self):
        entry = self._facts_entry(titles=("Film A",), year=None, runtime=None, genres=[], description="")
        messages = _build_chat_messages(
            history=[entry], system_prompt="SYS", current_content="now",
            max_turns=10, max_chars=1000,
        )
        assert messages[1]["content"] == "Here you go\n\n(Recommended movies:\n1. Film A)"

    def test_none_runtime_alone_is_dropped(self):
        entry = self._facts_entry(titles=("Film A",), runtime=None)
        messages = _build_chat_messages(
            history=[entry], system_prompt="SYS", current_content="now",
            max_turns=10, max_chars=1000,
        )
        assert "None" not in messages[1]["content"]
        assert "1. Film A (1999) — Drama, Crime: About Film A." in messages[1]["content"]

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


class TestStreamExplanationUnverified:
    @staticmethod
    async def _last_turn(**kwargs):
        mock_client = _mock_streaming_client([json.dumps({"message": {"content": "x"}})])
        movies = [{"serial_name": "Test Film", "genres": [], "description": ""}]
        with patch("core.ollama_client.httpx.AsyncClient", return_value=mock_client):
            _ = [t async for t in astream_explanation("q", movies, **kwargs)]
        return mock_client.stream.call_args.kwargs["json"]["messages"][-1]["content"]

    @pytest.mark.asyncio
    async def test_turn_names_the_conditions_the_catalog_could_not_check(self):
        turn = await self._last_turn(unverified=["silent", "with Keanu Reeves"])
        assert 'The catalog cannot check: "silent", "with Keanu Reeves".' in turn
        assert "Do not claim the movies meet it; say it was not checked." in turn

    @pytest.mark.asyncio
    async def test_turn_has_no_such_line_by_default(self):
        assert "cannot check" not in await self._last_turn()
        assert "cannot check" not in await self._last_turn(unverified=[])

    @pytest.mark.asyncio
    async def test_parts_stream_forwards_unverified(self):
        seen = {}

        async def fake(query, movies, history=None, unverified=None):
            seen["unverified"] = unverified
            yield "x"

        with patch("core.ollama_client.astream_explanation", fake):
            _ = [p async for p in astream_explanation_parts("q", [{"serial_name": "A"}], unverified=["silent"])]
        assert seen["unverified"] == ["silent"]


class TestConversationalComplaintRule:
    def test_rule_acknowledges_complaints_and_fills_the_unchecked_list(self):
        assert "{unchecked}" not in CONVERSATIONAL_SYSTEM_PROMPT
        assert settings.CATALOG_UNCHECKED_TEXT in CONVERSATIONAL_SYSTEM_PROMPT
        assert "complain" in CONVERSATIONAL_SYSTEM_PROMPT
        assert "one concrete next step" in CONVERSATIONAL_SYSTEM_PROMPT


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

    @pytest.mark.parametrize("key", [
        "history_max_turns", "history_max_chars_per_entry", "history_movie_description_chars",
    ])
    @pytest.mark.parametrize("bad", [0, -1, 1.5, "8", None, True])
    def test_rejects_values_below_one_or_non_int(self, key, bad):
        from recommender.settings import _validate_history_limits

        session = {
            "history_max_turns": 8, "history_max_chars_per_entry": 1000,
            "history_movie_description_chars": 200,
        }
        session[key] = bad
        with pytest.raises(ValueError, match=key):
            _validate_history_limits(session)

    def test_accepts_minimum_of_one(self):
        from recommender.settings import _validate_history_limits

        _validate_history_limits({
            "history_max_turns": 1, "history_max_chars_per_entry": 1,
            "history_movie_description_chars": 1,
        })


class TestNumCtxValidation:
    """params.yaml llm.num_ctx must be a positive int at settings load: it is
    sent as options.num_ctx on every call, so a bad value fails late or silently."""

    @pytest.mark.parametrize("bad", [0, -1, 1.5, "6144", None, True])
    def test_rejects_non_int_or_below_one(self, bad):
        from recommender.settings import _validate_num_ctx

        with pytest.raises(ValueError, match="llm.num_ctx"):
            _validate_num_ctx({"num_ctx": bad})

    def test_accepts_minimum_of_one(self):
        from recommender.settings import _validate_num_ctx

        _validate_num_ctx({"num_ctx": 1})

    def test_accepts_configured_value(self):
        from recommender.settings import _validate_num_ctx

        _validate_num_ctx({"num_ctx": 6144})


class TestOllamaOptions:
    """Every LLM call must carry the same options.num_ctx -- a differing value
    makes Ollama reload the model -- plus the fixed seed and the temperature
    for its kind of call: structured (JSON) calls are deterministic,
    generation calls keep some variety."""

    @staticmethod
    def _structured(settings):
        return {
            "num_ctx": settings.OLLAMA_NUM_CTX,
            "seed": settings.OLLAMA_SEED,
            "temperature": settings.OLLAMA_TEMPERATURE["structured"],
        }

    @staticmethod
    def _generation(settings):
        return {
            "num_ctx": settings.OLLAMA_NUM_CTX,
            "seed": settings.OLLAMA_SEED,
            "temperature": settings.OLLAMA_TEMPERATURE["generation"],
        }

    def test_options_by_kind(self, settings):
        settings.OLLAMA_NUM_CTX = 4096
        settings.OLLAMA_SEED = 7
        settings.OLLAMA_TEMPERATURE = {"structured": 0.0, "generation": 0.7}
        assert _ollama_options("structured") == {"num_ctx": 4096, "seed": 7, "temperature": 0.0}
        assert _ollama_options("generation") == {"num_ctx": 4096, "seed": 7, "temperature": 0.7}

    def test_rejects_unknown_kind(self):
        with pytest.raises(KeyError):
            _ollama_options("creative")

    def test_intent_payload_is_structured(self, settings):
        assert _intent_payload("a thriller")["options"] == self._structured(settings)

    def test_classify_and_parse_payload_is_structured(self, settings):
        payload = _classify_and_parse_payload("hi", [])
        assert payload["options"] == self._structured(settings)

    def test_explanation_payload_is_generation(self, settings):
        for stream in (False, True):
            payload = _explanation_payload("q", [], stream=stream)
            assert payload["options"] == self._generation(settings)

    def test_stream_chat_payload_is_generation(self, settings):
        payload = _stream_chat_payload([{"role": "user", "content": "hi"}])
        assert payload["options"] == self._generation(settings)

    def test_chat_sync_picks_kind_from_schema(self, settings):
        response = MagicMock()
        response.json.return_value = {"message": {"content": "ok"}}
        with patch("core.ollama_client.httpx.post", return_value=response) as post:
            assert _chat_sync([{"role": "user", "content": "hi"}], call_type="explanation") == "ok"
            _chat_sync([{"role": "user", "content": "hi"}], call_type="judge", schema={"type": "object"})
        plain, structured = post.call_args_list
        assert plain.kwargs["json"]["options"] == self._generation(settings)
        assert structured.kwargs["json"]["options"] == self._structured(settings)

    def test_num_ctx_identical_across_kinds(self):
        assert _ollama_options("structured")["num_ctx"] == _ollama_options("generation")["num_ctx"]


class TestSamplingValidation:
    """llm.temperature / llm.seed are checked at settings load, like num_ctx."""

    @staticmethod
    def _llm(**overrides):
        llm = {"temperature": {"structured": 0.0, "generation": 0.7}, "seed": 42}
        llm.update(overrides)
        return llm

    def test_accepts_configured_values(self):
        from recommender.settings import _validate_sampling

        _validate_sampling(self._llm())
        _validate_sampling(self._llm(temperature={"structured": 0, "generation": 2}))

    @pytest.mark.parametrize("bad", [-0.1, 2.1, "0.7", None, True])
    @pytest.mark.parametrize("kind", ["structured", "generation"])
    def test_rejects_temperature_out_of_range_or_non_number(self, kind, bad):
        from recommender.settings import _validate_sampling

        temperature = {"structured": 0.0, "generation": 0.7}
        temperature[kind] = bad
        with pytest.raises(ValueError, match=f"llm.temperature.{kind}"):
            _validate_sampling(self._llm(temperature=temperature))

    @pytest.mark.parametrize("temperature", [{"structured": 0.0}, {"generation": 0.7}, 0.7])
    def test_rejects_missing_temperature_kind(self, temperature):
        from recommender.settings import _validate_sampling

        with pytest.raises(ValueError, match="llm.temperature"):
            _validate_sampling(self._llm(temperature=temperature))

    @pytest.mark.parametrize("bad", [4.2, "42", None, True])
    def test_rejects_non_int_seed(self, bad):
        from recommender.settings import _validate_sampling

        with pytest.raises(ValueError, match="llm.seed"):
            _validate_sampling(self._llm(seed=bad))


class TestConversationalPromptGrounding:
    def test_prompt_makes_listed_facts_the_source_of_truth_and_flags_missing_details(self):
        from core.ollama_client import CONVERSATIONAL_SYSTEM_PROMPT

        assert "Recommended movies" in CONVERSATIONAL_SYSTEM_PROMPT
        assert "source of truth" in CONVERSATIONAL_SYSTEM_PROMPT
        assert "not in your catalog data" in CONVERSATIONAL_SYSTEM_PROMPT


class TestStructuredOutputSchema:
    """`format` carries a JSON schema (Ollama structured outputs), not the bare
    "json" mode. Pydantic validation and the repair retry still guard the reply."""

    def test_classify_payload_carries_the_message_intent_schema(self):
        payload = _classify_and_parse_payload("hi", [])
        assert payload["format"] is CLASSIFY_PARSE_SCHEMA
        assert payload["format"] != "json"

    def test_schema_has_category_enum_and_every_field_required(self):
        schema = CLASSIFY_PARSE_SCHEMA
        assert schema["type"] == "object"
        assert schema["properties"]["category"]["enum"] == [
            "new_search", "follow_up", "refinement", "general_chat", "clarify",
        ]
        assert set(schema["required"]) == set(MessageIntent.model_fields)
        assert set(schema["properties"]) == set(MessageIntent.model_fields)

    def test_schema_types_follow_the_model(self):
        props = CLASSIFY_PARSE_SCHEMA["properties"]
        assert props["genres"] == {"type": "array", "items": {"type": "string"}}
        assert {"type": "null"} in props["min_vote_average"]["anyOf"]
        assert {"type": "number"} in props["min_vote_average"]["anyOf"]
        assert {"type": "integer"} in props["max_runtime"]["anyOf"]

    def test_schema_uses_only_keywords_ollamas_grammar_handles(self):
        def keys(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    yield key
                    if key != "properties":
                        yield from keys(value)
                    else:
                        for prop in value.values():
                            yield from keys(prop)
            elif isinstance(node, list):
                for item in node:
                    yield from keys(item)

        assert set(keys(CLASSIFY_PARSE_SCHEMA)) <= {
            "type", "properties", "required", "enum", "items", "anyOf",
        }

    def test_schema_is_json_serializable(self):
        json.dumps(CLASSIFY_PARSE_SCHEMA)
        json.dumps(INTENT_SCHEMA)

    def test_intent_payload_schema_matches_what_extract_intent_reads(self):
        payload = _intent_payload("a comedy")
        assert payload["format"] is INTENT_SCHEMA
        assert "category" not in INTENT_SCHEMA["properties"]
        assert "semantic_query" not in INTENT_SCHEMA["properties"]
        assert set(INTENT_SCHEMA["required"]) == set(_extract_intent({}))

    def test_message_intent_validates_a_reply_shaped_by_the_schema(self):
        reply = {
            "category": "new_search", "semantic_query": "q", "standalone_query": "q",
            "genres": ["Comedy"], "mood": "",
            "themes": [], "negations": [], "reference_films": [], "country_exclusions": [],
            "country_inclusions": [], "min_vote_average": None, "min_release_year": None,
            "max_release_year": None, "min_runtime": None, "max_runtime": 90,
            "original_languages": [], "unsupported_constraints": [], "result_count": None,
        }
        assert set(reply) == set(CLASSIFY_PARSE_SCHEMA["properties"])
        assert MessageIntent.model_validate(reply).max_runtime == 90

    @pytest.mark.asyncio
    async def test_request_sends_the_schema_on_the_wire(self):
        mock_client = _mock_ollama_responses('{"category": "new_search"}')
        with patch("core.ollama_client.httpx.AsyncClient", return_value=mock_client):
            await aclassify_and_parse("hi")
        assert mock_client.post.call_args.kwargs["json"]["format"] == CLASSIFY_PARSE_SCHEMA

    @pytest.mark.asyncio
    async def test_repair_retry_still_happens_with_the_schema(self):
        mock_client = _mock_ollama_responses(
            '{"category": "nonsense"}', '{"category": "follow_up", "semantic_query": "x"}'
        )
        with patch("core.ollama_client.httpx.AsyncClient", return_value=mock_client):
            intent = await aclassify_and_parse("tell me about it")
        assert intent.category == "follow_up"
        formats = [c.kwargs["json"]["format"] for c in mock_client.post.call_args_list]
        assert formats == [CLASSIFY_PARSE_SCHEMA, CLASSIFY_PARSE_SCHEMA]

    def test_chat_sync_sends_the_schema_as_format(self):
        response = MagicMock()
        response.json.return_value = {"message": {"content": "{}"}}
        schema = {"type": "object", "properties": {"score": {"type": "integer"}}}
        with patch("core.ollama_client.httpx.post", return_value=response) as post:
            _chat_sync([{"role": "user", "content": "hi"}], call_type="judge", schema=schema)
        assert post.call_args.kwargs["json"]["format"] == schema

    def test_chat_sync_without_schema_sends_no_format(self):
        response = MagicMock()
        response.json.return_value = {"message": {"content": "ok"}}
        with patch("core.ollama_client.httpx.post", return_value=response) as post:
            _chat_sync([{"role": "user", "content": "hi"}], call_type="explanation")
        assert "format" not in post.call_args.kwargs["json"]

    def test_parse_intent_sends_the_intent_schema(self):
        response = MagicMock()
        response.json.return_value = {"message": {"content": '{"genres": ["Comedy"]}'}}
        with patch("core.ollama_client.httpx.post", return_value=response) as post:
            parse_intent("a comedy")
        assert post.call_args.kwargs["json"]["format"] == INTENT_SCHEMA

    def test_strip_think_still_runs_before_json_parsing(self):
        response = MagicMock()
        response.json.return_value = {
            "message": {"content": '<think>hmm</think>{"genres": ["Comedy"]}'}
        }
        with patch("core.ollama_client.httpx.post", return_value=response), \
             patch("core.ollama_client._resolve_fields", side_effect=_unresolved):
            assert parse_intent("a comedy")["genres"] == ["Comedy"]


FINAL_CHUNK = {
    "done": True,
    "prompt_eval_count": 812,
    "eval_count": 96,
    "prompt_eval_duration": 143_200_000,
    "eval_duration": 1_021_700_000,
}


def _call_records(caplog):
    return [r for r in caplog.records if r.getMessage().startswith("Ollama call=")]


class TestCallLogging:
    """One INFO line per Ollama call with tokens and Ollama's own timings."""

    def test_line_has_call_type_counts_and_durations_in_ms(self, caplog):
        with caplog.at_level(logging.INFO, logger="core.ollama_client"):
            _log_call_stats("classify_parse", FINAL_CHUNK)
        (record,) = _call_records(caplog)
        message = record.getMessage()
        assert record.levelno == logging.INFO
        assert "call=classify_parse" in message
        assert "prompt_tokens=812" in message
        assert "eval_tokens=96" in message
        assert "prompt_eval_ms=143 " in message
        assert "eval_ms=1022 " in message
        assert "ttft_ms=n/a" in message

    def test_ttft_is_reported_when_measured(self, caplog):
        with caplog.at_level(logging.INFO, logger="core.ollama_client"):
            _log_call_stats("explanation", FINAL_CHUNK, ttft_ms=250.4)
        assert "ttft_ms=250" in _call_records(caplog)[0].getMessage()

    @pytest.mark.parametrize("stats", [{}, None, "garbage", {"prompt_eval_count": None}])
    def test_missing_or_malformed_fields_do_not_crash(self, caplog, stats):
        with caplog.at_level(logging.INFO, logger="core.ollama_client"):
            _log_call_stats("judge", stats)
        (record,) = _call_records(caplog)
        assert "prompt_tokens=n/a eval_tokens=n/a prompt_eval_ms=n/a eval_ms=n/a" in record.getMessage()

    def test_bool_counts_are_not_mistaken_for_numbers(self, caplog):
        with caplog.at_level(logging.INFO, logger="core.ollama_client"):
            _log_call_stats("judge", {"prompt_eval_count": True, "prompt_eval_duration": True})
        assert "prompt_tokens=n/a" in _call_records(caplog)[0].getMessage()
        assert "prompt_eval_ms=n/a" in _call_records(caplog)[0].getMessage()

    def test_warns_when_prompt_is_within_ten_percent_of_num_ctx(self, caplog, settings):
        settings.OLLAMA_NUM_CTX = 1000
        with caplog.at_level(logging.INFO, logger="core.ollama_client"):
            _log_call_stats("conversational", {"prompt_eval_count": 900})
        (warning,) = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert "900 of 1000" in warning.getMessage()
        assert "call=conversational" in warning.getMessage()

    def test_no_warning_below_the_threshold(self, caplog, settings):
        settings.OLLAMA_NUM_CTX = 1000
        with caplog.at_level(logging.INFO, logger="core.ollama_client"):
            _log_call_stats("conversational", {"prompt_eval_count": 899})
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_no_warning_without_a_prompt_count(self, caplog):
        with caplog.at_level(logging.INFO, logger="core.ollama_client"):
            _log_call_stats("conversational", {})
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_chat_sync_logs_with_its_call_type(self, caplog):
        response = MagicMock()
        response.json.return_value = {"message": {"content": "ok"}, **FINAL_CHUNK}
        with patch("core.ollama_client.httpx.post", return_value=response), \
             caplog.at_level(logging.INFO, logger="core.ollama_client"):
            _chat_sync([{"role": "user", "content": "hi"}], call_type="judge")
        assert "call=judge prompt_tokens=812" in _call_records(caplog)[0].getMessage()

    @pytest.mark.parametrize("call, call_type", [
        (lambda: parse_intent("q"), "intent"),
    ])
    def test_sync_wrappers_label_their_calls(self, caplog, call, call_type):
        response = MagicMock()
        response.json.return_value = {"message": {"content": "{}"}, **FINAL_CHUNK}
        with patch("core.ollama_client.httpx.post", return_value=response), \
             patch("core.ollama_client._resolve_fields", side_effect=_unresolved), \
             caplog.at_level(logging.INFO, logger="core.ollama_client"):
            call()
        assert f"call={call_type} " in _call_records(caplog)[0].getMessage()

    @pytest.mark.asyncio
    async def test_classify_parse_logs_each_attempt(self, caplog):
        body = {"message": {"content": '{"category": "new_search"}'}, **FINAL_CHUNK}
        mock_client = _mock_ollama_responses("x")
        mock_client.post.side_effect = [MagicMock(json=MagicMock(return_value=body))]
        with patch("core.ollama_client.httpx.AsyncClient", return_value=mock_client), \
             caplog.at_level(logging.INFO, logger="core.ollama_client"):
            await aclassify_and_parse("hi")
        (record,) = _call_records(caplog)
        assert "call=classify_parse prompt_tokens=812" in record.getMessage()

    @pytest.mark.asyncio
    async def test_classify_parse_response_without_stats_still_parses(self, caplog):
        mock_client = _mock_ollama_responses('{"category": "general_chat"}')
        with patch("core.ollama_client.httpx.AsyncClient", return_value=mock_client), \
             caplog.at_level(logging.INFO, logger="core.ollama_client"):
            intent = await aclassify_and_parse("hi")
        assert intent.category == "general_chat"
        assert "prompt_tokens=n/a" in _call_records(caplog)[0].getMessage()


class TestStreamCallLogging:
    @staticmethod
    def _lines(*chunks):
        return [json.dumps(c) for c in chunks]

    def test_stream_stats_takes_final_chunk_and_first_content_time(self):
        stats = _StreamStats()
        stats.observe({"message": {"content": ""}, "done": False})
        assert stats.ttft_ms is None
        stats.observe({"message": {"content": "Hi"}, "done": False})
        first = stats.ttft_ms
        assert first is not None and first >= 0
        stats.observe({"message": {"content": " there"}, "done": False})
        assert stats.ttft_ms == first
        stats.observe({"message": {"content": ""}, **FINAL_CHUNK})
        assert stats.final["eval_count"] == 96

    @pytest.mark.asyncio
    async def test_conversational_stream_logs_final_chunk_and_ttft(self, caplog):
        lines = self._lines(
            {"message": {"content": "Sure"}, "done": False},
            {"message": {"content": ""}, **FINAL_CHUNK},
        )
        with patch("core.ollama_client.httpx.AsyncClient", return_value=_mock_streaming_client(lines)), \
             caplog.at_level(logging.INFO, logger="core.ollama_client"):
            tokens = [t async for t in astream_conversational("hi")]
        assert tokens == ["Sure"]
        (record,) = _call_records(caplog)
        message = record.getMessage()
        assert "call=conversational prompt_tokens=812 eval_tokens=96" in message
        assert "ttft_ms=n/a" not in message

    @pytest.mark.asyncio
    async def test_explanation_stream_logs_with_its_call_type(self, caplog):
        lines = self._lines({"message": {"content": "Opening"}, **FINAL_CHUNK})
        with patch("core.ollama_client.httpx.AsyncClient", return_value=_mock_streaming_client(lines)), \
             caplog.at_level(logging.INFO, logger="core.ollama_client"):
            _ = [t async for t in astream_explanation("q", [])]
        assert "call=explanation " in _call_records(caplog)[0].getMessage()

    @pytest.mark.asyncio
    async def test_stream_without_a_final_chunk_still_streams_and_logs_na(self, caplog):
        lines = self._lines({"message": {"content": "Hello"}})
        with patch("core.ollama_client.httpx.AsyncClient", return_value=_mock_streaming_client(lines)), \
             caplog.at_level(logging.INFO, logger="core.ollama_client"):
            tokens = [t async for t in astream_conversational("hi")]
        assert tokens == ["Hello"]
        assert "prompt_tokens=n/a eval_tokens=n/a" in _call_records(caplog)[0].getMessage()

    def test_sync_stream_logs(self, caplog):
        from core.ollama_client import stream_explanation

        class _Response:
            def iter_lines(self):
                yield json.dumps({"message": {"content": "Hi"}, **FINAL_CHUNK})

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        with patch("core.ollama_client.httpx.stream", return_value=_Response()), \
             caplog.at_level(logging.INFO, logger="core.ollama_client"):
            assert "".join(stream_explanation("q", [])) == "Hi"
        assert "call=explanation prompt_tokens=812" in _call_records(caplog)[0].getMessage()


TAGS = {
    "models": [
        {"name": "other:1b", "digest": "sha256:aaa"},
        {"name": "qwen3:8b", "digest": "sha256:bbb"},
    ]
}


class TestModelDigestLogging:
    @pytest.fixture(autouse=True)
    def _lookup_enabled(self, monkeypatch):
        monkeypatch.setattr("core.ollama_client._digest_lookup_attempted", False)

    @staticmethod
    def _digest_lines(caplog):
        return [r.getMessage() for r in caplog.records if "digest" in r.getMessage()]

    def test_logs_the_digest_of_the_configured_model(self, caplog, settings):
        settings.OLLAMA_MODEL = "qwen3:8b"
        response = MagicMock()
        response.json.return_value = TAGS
        with patch("core.ollama_client.httpx.get", return_value=response) as get, \
             caplog.at_level(logging.INFO, logger="core.ollama_client"):
            _log_model_digest_once()
        assert get.call_args.args[0].endswith("/api/tags")
        assert self._digest_lines(caplog) == ["Ollama model qwen3:8b digest sha256:bbb"]

    def test_only_one_lookup_per_process(self):
        response = MagicMock()
        response.json.return_value = TAGS
        with patch("core.ollama_client.httpx.get", return_value=response) as get:
            _log_model_digest_once()
            _log_model_digest_once()
        assert get.call_count == 1

    def test_failure_is_tolerated_and_not_retried(self, caplog):
        with patch("core.ollama_client.httpx.get", side_effect=httpx.ConnectError("down")) as get, \
             caplog.at_level(logging.INFO, logger="core.ollama_client"):
            _log_model_digest_once()
            _log_model_digest_once()
        assert get.call_count == 1
        assert any("Could not read the Ollama model digest" in r.getMessage() for r in caplog.records)

    def test_unexpected_body_is_tolerated(self):
        response = MagicMock()
        response.json.return_value = ["not", "a", "dict"]
        with patch("core.ollama_client.httpx.get", return_value=response):
            _log_model_digest_once()

    @pytest.mark.parametrize(
        "body",
        [
            {"models": None},
            ["not", "a", "dict"],
            {"models": "qwen3:8b"},
            {"models": ["qwen3:8b", None, 3]},
            {"models": [{"name": 7, "digest": "sha256:zzz"}, {"name": None}]},
            {"models": [{"name": ["qwen3:8b"], "digest": "sha256:zzz"}]},
            {"models": [{"name": {"a": 1}, "digest": "sha256:zzz"}]},
        ],
    )
    def test_malformed_tags_bodies_never_raise(self, body, caplog):
        response = MagicMock()
        response.json.return_value = body
        with patch("core.ollama_client.httpx.get", return_value=response), \
             caplog.at_level(logging.INFO, logger="core.ollama_client"):
            _log_model_digest_once()
        assert any("digest unknown" in r.getMessage() for r in caplog.records)

    def test_any_exception_in_the_lookup_is_swallowed(self):
        with patch("core.ollama_client.httpx.get", side_effect=TypeError("boom")):
            _log_model_digest_once()

    def test_model_missing_from_tags_is_reported(self, caplog, settings):
        settings.OLLAMA_MODEL = "absent:1b"
        response = MagicMock()
        response.json.return_value = TAGS
        with patch("core.ollama_client.httpx.get", return_value=response), \
             caplog.at_level(logging.INFO, logger="core.ollama_client"):
            _log_model_digest_once()
        assert any("digest unknown" in r.getMessage() for r in caplog.records)

    def test_untagged_model_name_matches_latest(self, caplog, settings):
        settings.OLLAMA_MODEL = "llama3"
        response = MagicMock()
        response.json.return_value = {"models": [{"name": "llama3:latest", "digest": "sha256:ccc"}]}
        with patch("core.ollama_client.httpx.get", return_value=response), \
             caplog.at_level(logging.INFO, logger="core.ollama_client"):
            _log_model_digest_once()
        assert self._digest_lines(caplog) == ["Ollama model llama3 digest sha256:ccc"]

    @pytest.mark.asyncio
    async def test_async_variant_logs_and_runs_once(self, caplog, settings):
        settings.OLLAMA_MODEL = "qwen3:8b"
        client = AsyncMock()
        client.get.return_value = MagicMock(json=MagicMock(return_value=TAGS))
        with caplog.at_level(logging.INFO, logger="core.ollama_client"):
            await _alog_model_digest_once(client)
            await _alog_model_digest_once(client)
        assert client.get.await_count == 1
        assert self._digest_lines(caplog) == ["Ollama model qwen3:8b digest sha256:bbb"]

    @pytest.mark.asyncio
    async def test_async_failure_is_tolerated(self):
        client = AsyncMock()
        client.get.side_effect = httpx.ConnectError("down")
        await _alog_model_digest_once(client)

    @pytest.mark.asyncio
    async def test_async_any_exception_in_the_lookup_is_swallowed(self):
        client = AsyncMock()
        client.get.side_effect = TypeError("boom")
        await _alog_model_digest_once(client)

    @pytest.mark.asyncio
    async def test_classify_still_parses_when_the_digest_lookup_blows_up(self):
        mock_client = _mock_ollama_responses('{"category": "general_chat"}')
        mock_client.get = AsyncMock(
            return_value=MagicMock(json=MagicMock(return_value={"models": None}))
        )
        with patch("core.ollama_client.httpx.AsyncClient", return_value=mock_client):
            result = await aclassify_and_parse("hi")
        assert result is not None
        assert mock_client.get.await_count == 1

    @pytest.mark.asyncio
    async def test_first_chat_request_triggers_the_lookup(self):
        mock_client = _mock_ollama_responses('{"category": "general_chat"}')
        mock_client.get = AsyncMock(return_value=MagicMock(json=MagicMock(return_value=TAGS)))
        with patch("core.ollama_client.httpx.AsyncClient", return_value=mock_client):
            await aclassify_and_parse("hi")
        assert mock_client.get.await_count == 1


class TestExplanationDescriptionCap:
    MOVIES = [{"serial_name": "Alien", "genres": ["Horror"], "description": "x" * 500}]

    def test_cap_comes_from_settings(self, settings):
        settings.OLLAMA_EXPLANATION_DESCRIPTION_CHARS = 7
        context = _build_movies_context(self.MOVIES)
        assert "x" * 7 in context and "x" * 8 not in context

    def test_default_cap_is_the_params_value(self, settings):
        assert settings.OLLAMA_EXPLANATION_DESCRIPTION_CHARS == 200
        context = _build_movies_context(self.MOVIES)
        assert "x" * 200 in context and "x" * 201 not in context

    @pytest.mark.parametrize("bad", [0, -1, 1.5, "200", None, True])
    def test_rejects_non_positive_or_non_int(self, bad):
        from recommender.settings import _validate_explanation_description_chars

        with pytest.raises(ValueError, match="llm.explanation_description_chars"):
            _validate_explanation_description_chars({"explanation_description_chars": bad})

    def test_accepts_a_positive_int(self):
        from recommender.settings import _validate_explanation_description_chars

        _validate_explanation_description_chars({"explanation_description_chars": 1})


class TestCatalogYearRule:
    """The classify prompt tells the model where the catalog ends, so "newest"
    never turns into the calendar year."""

    BOUNDS = (1915, 2023, datetime.date(2023, 9, 30))

    def _system_prompt(self, bounds):
        return _classify_and_parse_payload("newest movies", [], year_bounds=bounds)["messages"][0]["content"]

    def test_rule_names_the_catalog_max_year_and_the_recent_window(self):
        prompt = self._system_prompt(self.BOUNDS)
        assert "up to 2023" in prompt
        assert "never the calendar year" in prompt
        assert f"{2023 - settings.INTENT_RECENT_YEARS + 1}" in prompt

    def test_rule_is_absent_when_bounds_are_none(self):
        prompt = self._system_prompt(None)
        assert "never the calendar year" not in prompt

    def test_prompt_formats_with_no_stray_placeholders(self):
        for bounds in (self.BOUNDS, None):
            assert "{year_rule}" not in self._system_prompt(bounds)
            assert "{catalog_max_year}" not in self._system_prompt(bounds)

    def test_default_payload_omits_the_rule(self):
        assert "never the calendar year" not in _classify_and_parse_payload("hi", [])["messages"][0]["content"]

    @pytest.mark.asyncio
    async def test_request_passes_the_catalog_bounds_into_the_prompt(self):
        mock_client = _mock_ollama_responses(
            json.dumps({"category": "new_search", "semantic_query": "x", "standalone_query": "x"})
        )
        with patch("core.ollama_client.httpx.AsyncClient", return_value=mock_client), \
                patch("core.ollama_client.catalog_year_range", return_value=self.BOUNDS):
            await aclassify_and_parse("newest movies")
        sent = mock_client.post.call_args.kwargs["json"]["messages"][0]["content"]
        assert "up to 2023" in sent
