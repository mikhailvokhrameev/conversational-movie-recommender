import inspect
import json
import logging
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
from django.conf import settings
from django.test import AsyncRequestFactory

from core.ollama_client import CLARIFY_SYSTEM_PROMPT, NON_FILTER_FIELDS, MessageIntent, aclassify_and_parse, astream_explanation_parts
from movies.views import (
    NO_MATCH_MESSAGE, ChatView, _assistant_turn_text, _describe_filters, _first_line, _generate_and_score,
    _no_match_message, _unchecked_constraints,
)


def _mock_classify_and_parse(category, semantic_query="", standalone_query="", **filters):
    intent = MessageIntent(
        category=category, semantic_query=semantic_query,
        standalone_query=standalone_query or semantic_query, **filters,
    )

    async def _classify(message, history=None):
        return intent
    return _classify


def _mock_encode(embedding=None):
    return embedding or [0.1] * 1024


def _mock_generate_and_score(movies=None):
    async def _gen(query_embedding, intent, query_text="", exclude_ids=None):
        result = movies or [
            {
                "id": 1, "tmdb_id": 1, "serial_name": "Test Film", "original_title": "Test Film",
                "genres": ["Drama"], "country": ["US"], "release_date": "2024-01-01",
                "description": "A test", "runtime": 120, "vote_average": 7.5,
                "poster_path": "/abc123.jpg",
                "embedding": [0.1] * 1024, "total": 0.85, "semantic": 0.4,
                "metadata": 0.3, "session": 0.15,
            }
        ]
        return result, result
    return _gen


def _two_movies():
    return [
        {
            "id": i, "tmdb_id": i, "serial_name": f"Film {i}", "original_title": f"Film {i}",
            "genres": ["Comedy"], "country": ["US"], "release_date": "2024-01-01",
            "description": "A test", "runtime": 120, "vote_average": 7.5,
            "poster_path": "/abc123.jpg",
            "embedding": [0.1] * 1024, "total": 0.85, "semantic": 0.4,
            "metadata": 0.3, "session": 0.15,
        }
        for i in (1, 2)
    ]


def _parse_sse(text):
    """Split an SSE body into (event, parsed data) pairs."""
    events = []
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        if len(lines) == 2 and lines[0].startswith("event: "):
            events.append((lines[0][len("event: "):], json.loads(lines[1][len("data: "):])))
    return events


async def _empty_stream(*args, **kwargs):
    return
    yield


async def _empty_parts_stream(query, movies, history=None):
    """Stand-in for astream_explanation_parts: same (slot, text) contract, no output."""
    return
    yield


async def _one_token_stream(*args, **kwargs):
    yield "ответ"


def _mock_explanation_parts(*parts):
    """Stand-in for astream_explanation_parts yielding the given (slot, text) pairs."""
    async def _stream(query, movies, history=None):
        for part in parts:
            yield part
    return _stream


class TestMockContracts:
    """The SSE tests patch out _generate_and_score. ChatView catches every
    exception from the pipeline and turns it into a 500, so a mock whose
    signature has drifted from the real function shows up as an opaque
    `assert 500 == 200` rather than a TypeError. Pin the signature here so
    drift fails with a message that names the problem."""

    def test_generate_and_score_mock_matches_real_signature(self):
        real = inspect.signature(_generate_and_score)
        mock = inspect.signature(_mock_generate_and_score())
        assert list(mock.parameters) == list(real.parameters), (
            "_mock_generate_and_score is out of sync with movies.views."
            "_generate_and_score — update the mock to match the real signature"
        )

    def test_classify_and_parse_mock_matches_real_signature(self):
        real = inspect.signature(aclassify_and_parse)
        mock = inspect.signature(_mock_classify_and_parse("new_search"))
        assert list(mock.parameters) == list(real.parameters), (
            "_mock_classify_and_parse is out of sync with core.ollama_client."
            "aclassify_and_parse — update the mock to match the real signature"
        )

    @pytest.mark.parametrize("mock", [_empty_parts_stream, _mock_explanation_parts()])
    def test_explanation_parts_mocks_match_real_signature(self, mock):
        real = inspect.signature(astream_explanation_parts)
        assert list(inspect.signature(mock).parameters) == list(real.parameters), (
            "an explanation-parts stub is out of sync with core.ollama_client."
            "astream_explanation_parts — update the stub to match the real signature"
        )


@pytest.mark.django_db
class TestChatSSENewSearch:
    @pytest.mark.asyncio
    async def test_new_search_returns_movies_and_tokens(self):
        factory = AsyncRequestFactory()
        request = factory.post(
            "/api/chat/",
            data=json.dumps({"message": "хочу комедию"}),
            content_type="application/json",
        )

        append_history_mock = AsyncMock()
        with patch("movies.views.aclassify_and_parse", _mock_classify_and_parse("new_search", semantic_query="хочу комедию")), \
             patch("movies.views.encode_query", return_value=[0.1] * 1024), \
             patch("movies.views._generate_and_score", _mock_generate_and_score()), \
             patch("movies.views._save_session", AsyncMock()), \
             patch("movies.views._append_history", append_history_mock), \
             patch("movies.views.astream_explanation_parts", _mock_explanation_parts((None, "Вот подборка."))):

            view = ChatView.as_view()
            response = await view(request)

            assert response.status_code == 200
            assert response["Content-Type"] == "text/event-stream"

            body = b""
            async for chunk in response.streaming_content:
                body += chunk if isinstance(chunk, bytes) else chunk.encode()

            text = body.decode()
            assert "event: movies" in text
            assert "event: done" in text
            assert "Test Film" in text

        append_history_mock.assert_awaited_once()
        call = append_history_mock.await_args
        assert call.args[1] == "assistant"
        assert call.kwargs["movies"] == ["Test Film"]
        assert call.kwargs["count_turn"] is False
        assert call.kwargs["movie_facts"] == [{
            "title": "Test Film", "year": 2024, "genres": ["Drama"], "runtime": 120,
            "description": "A test",
        }]

    @pytest.mark.asyncio
    async def test_stored_movie_facts_truncate_description_to_configured_cap(self, settings):
        settings.SESSION_HISTORY_MOVIE_DESCRIPTION_CHARS = 10
        long_movie = _two_movies()[:1]
        long_movie[0]["description"] = "x" * 500
        request = AsyncRequestFactory().post(
            "/api/chat/",
            data=json.dumps({"message": "хочу комедию"}),
            content_type="application/json",
        )
        append_history_mock = AsyncMock()
        with patch("movies.views.aclassify_and_parse", _mock_classify_and_parse("new_search", semantic_query="комедия")), \
             patch("movies.views.encode_query", return_value=[0.1] * 1024), \
             patch("movies.views._generate_and_score", _mock_generate_and_score(long_movie)), \
             patch("movies.views._save_session", AsyncMock()), \
             patch("movies.views._append_history", append_history_mock), \
             patch("movies.views.astream_explanation_parts", _mock_explanation_parts((None, "Вот подборка."))):
            response = await ChatView.as_view()(request)
            async for _ in response.streaming_content:
                pass

        assert append_history_mock.await_args.kwargs["movie_facts"][0]["description"] == "x" * 10

    @pytest.mark.asyncio
    async def test_search_uses_standalone_query_as_query_text(self):
        factory = AsyncRequestFactory()
        request = factory.post(
            "/api/chat/",
            data=json.dumps({"message": "suggest something similar"}),
            content_type="application/json",
        )
        seen = {}
        inner = _mock_generate_and_score()

        async def _gen(query_embedding, intent, query_text="", exclude_ids=None):
            seen["query_text"] = query_text
            return await inner(query_embedding, intent, query_text)

        classify = _mock_classify_and_parse(
            "new_search", semantic_query="crime drama mafia",
            standalone_query="crime dramas similar to The Godfather",
        )
        with patch("movies.views.aclassify_and_parse", classify), \
             patch("movies.views.encode_query", return_value=[0.1] * 1024), \
             patch("movies.views._generate_and_score", _gen), \
             patch("movies.views._save_session", AsyncMock()), \
             patch("movies.views._append_history", AsyncMock()), \
             patch("movies.views.astream_explanation_parts", _empty_parts_stream):
            response = await ChatView.as_view()(request)
            async for _ in response.streaming_content:
                pass

        assert seen["query_text"] == "crime dramas similar to The Godfather"

    @pytest.mark.asyncio
    async def test_intent_event_has_no_query_fields(self):
        factory = AsyncRequestFactory()
        request = factory.post(
            "/api/chat/",
            data=json.dumps({"message": "хочу комедию"}),
            content_type="application/json",
        )
        with patch("movies.views.aclassify_and_parse", _mock_classify_and_parse("new_search", semantic_query="комедия")), \
             patch("movies.views.encode_query", return_value=[0.1] * 1024), \
             patch("movies.views._generate_and_score", _mock_generate_and_score()), \
             patch("movies.views._save_session", AsyncMock()), \
             patch("movies.views._append_history", AsyncMock()), \
             patch("movies.views.astream_explanation_parts", _empty_parts_stream):
            response = await ChatView.as_view()(request)
            body = b""
            async for chunk in response.streaming_content:
                body += chunk if isinstance(chunk, bytes) else chunk.encode()

        text = body.decode()
        data_line = text.split("event: movies\n", 1)[1].split("\n", 1)[0]
        intent = json.loads(data_line.removeprefix("data: "))["intent"]
        assert "semantic_query" not in intent
        assert "standalone_query" not in intent

    @pytest.mark.asyncio
    async def test_embeds_semantic_query_not_raw_message(self):
        factory = AsyncRequestFactory()
        request = factory.post(
            "/api/chat/",
            data=json.dumps({"message": "ну хочу такую весёлую комедию пожалуйста"}),
            content_type="application/json",
        )
        encode_mock = MagicMock(return_value=[0.1] * 1024)

        with patch("movies.views.aclassify_and_parse", _mock_classify_and_parse("new_search", semantic_query="весёлая комедия")), \
             patch("movies.views.encode_query", encode_mock), \
             patch("movies.views._generate_and_score", _mock_generate_and_score()), \
             patch("movies.views._save_session", AsyncMock()), \
             patch("movies.views._append_history", AsyncMock()), \
             patch("movies.views.astream_explanation_parts", _empty_parts_stream):

            view = ChatView.as_view()
            response = await view(request)
            async for _ in response.streaming_content:
                pass

            encode_mock.assert_called_once_with("весёлая комедия")

    @pytest.mark.asyncio
    async def test_intro_streams_as_token_and_descriptions_as_indexed_blurbs(self):
        """The opening line keeps the old `token` wire format so it still streams
        under the caret; each description is addressed to its movie's card."""
        factory = AsyncRequestFactory()
        request = factory.post(
            "/api/chat/",
            data=json.dumps({"message": "хочу комедию"}),
            content_type="application/json",
        )
        parts = _mock_explanation_parts(
            (None, "Вот подборка "),
            (None, "комедий."),
            (0, "Про первый фильм."),
            (1, "Про второй фильм."),
        )
        append_history_mock = AsyncMock()

        with patch("movies.views.aclassify_and_parse", _mock_classify_and_parse("new_search", semantic_query="хочу комедию")), \
             patch("movies.views.encode_query", return_value=[0.1] * 1024), \
             patch("movies.views._generate_and_score", _mock_generate_and_score(_two_movies())), \
             patch("movies.views._save_session", AsyncMock()), \
             patch("movies.views._append_history", append_history_mock), \
             patch("movies.views.astream_explanation_parts", parts):

            view = ChatView.as_view()
            response = await view(request)

            body = b""
            async for chunk in response.streaming_content:
                body += chunk if isinstance(chunk, bytes) else chunk.encode()

            events = _parse_sse(body.decode())

        assert [d["text"] for e, d in events if e == "token"] == [
            "Вот подборка ", "комедий."
        ]
        assert [(d["index"], d["text"]) for e, d in events if e == "blurb"] == [
            (0, "Про первый фильм."), (1, "Про второй фильм."),
        ]
        assert "error" not in [e for e, _ in events]
        # Markers arrived, so the intro already holds only the opening line.
        assert "intro_final" not in [e for e, _ in events]

        # Stored in the shape the model is asked for (intro, then each
        # description under its [[n]] marker), so later turns see the format.
        append_history_mock.assert_awaited_once()
        call = append_history_mock.await_args
        assert call.args[1] == "assistant"
        assert call.args[2] == "Вот подборка комедий.\n[[1]]\nПро первый фильм.\n[[2]]\nПро второй фильм."
        assert call.kwargs["movies"] == ["Film 1", "Film 2"]

    @pytest.mark.asyncio
    async def test_assistant_turn_keeps_markers_before_descriptions(self):
        factory = AsyncRequestFactory()
        request = factory.post(
            "/api/chat/",
            data=json.dumps({"message": "хочу комедию"}),
            content_type="application/json",
        )
        append_history_mock = AsyncMock()

        with patch("movies.views.aclassify_and_parse", _mock_classify_and_parse("new_search", semantic_query="хочу комедию")), \
             patch("movies.views.encode_query", return_value=[0.1] * 1024), \
             patch("movies.views._generate_and_score", _mock_generate_and_score()), \
             patch("movies.views._save_session", AsyncMock()), \
             patch("movies.views._append_history", append_history_mock), \
             patch("movies.views.astream_explanation_parts",
                   _mock_explanation_parts((None, "Intro."), (0, "Desc one."))):

            view = ChatView.as_view()
            response = await view(request)
            body = b""
            async for chunk in response.streaming_content:
                body += chunk if isinstance(chunk, bytes) else chunk.encode()
            events = _parse_sse(body.decode())

        assert [d["text"] for e, d in events if e == "token"] == ["Intro."]
        assert [(d["index"], d["text"]) for e, d in events if e == "blurb"] == [(0, "Desc one.")]
        append_history_mock.assert_awaited_once()
        assert append_history_mock.await_args.args[2] == "Intro.\n[[1]]\nDesc one."

    @pytest.mark.asyncio
    async def test_no_markers_sends_intro_final_and_stores_only_the_first_line(self):
        """The model wrote every description into the intro: the client swaps the
        streamed text for the opening line and the cards fall back to catalog copy."""
        factory = AsyncRequestFactory()
        request = factory.post(
            "/api/chat/",
            data=json.dumps({"message": "хочу комедию"}),
            content_type="application/json",
        )
        append_history_mock = AsyncMock()

        with patch("movies.views.aclassify_and_parse", _mock_classify_and_parse("new_search", semantic_query="хочу комедию")), \
             patch("movies.views.encode_query", return_value=[0.1] * 1024), \
             patch("movies.views._generate_and_score", _mock_generate_and_score(_two_movies())), \
             patch("movies.views._save_session", AsyncMock()), \
             patch("movies.views._append_history", append_history_mock), \
             patch("movies.views.astream_explanation_parts",
                   _mock_explanation_parts((None, "Интро.\nПро первый. "), (None, "Про второй."))):

            view = ChatView.as_view()
            response = await view(request)
            body = b""
            async for chunk in response.streaming_content:
                body += chunk if isinstance(chunk, bytes) else chunk.encode()
            events = _parse_sse(body.decode())

        names = [e for e, _ in events]
        assert [d for e, d in events if e == "intro_final"] == [{"text": "Интро."}]
        assert names.index("intro_final") < names.index("done")
        append_history_mock.assert_awaited_once()
        assert append_history_mock.await_args.args[2] == "Интро."


class TestAssistantTurnText:
    """The stored assistant turn mirrors the format the explanation prompt asks
    for; pure helpers, so they run without a database."""

    def test_intro_then_each_description_under_its_marker(self):
        assert _assistant_turn_text("Интро.\n", {0: ["Про ", "первый.\n"], 1: ["Про второй."]}) == (
            "Интро.\n[[1]]\nПро первый.\n[[2]]\nПро второй."
        )

    def test_slots_are_written_in_index_order(self):
        assert _assistant_turn_text("И.", {1: ["Два."], 0: ["Один."]}) == "И.\n[[1]]\nОдин.\n[[2]]\nДва."

    def test_missing_intro_starts_with_the_first_marker(self):
        assert _assistant_turn_text("", {0: ["Один."]}) == "[[1]]\nОдин."

    def test_nothing_to_store_is_empty(self):
        assert _assistant_turn_text("  ", {}) == ""

    def test_first_line_skips_leading_blank_lines(self):
        assert _first_line("\n\nИнтро.\nПро первый.\n\nПро второй.") == "Интро."


@pytest.mark.django_db
class TestChatSSEConversational:
    @pytest.mark.asyncio
    async def test_follow_up_returns_text_only(self):
        factory = AsyncRequestFactory()
        request = factory.post(
            "/api/chat/",
            data=json.dumps({"message": "расскажи подробнее"}),
            content_type="application/json",
        )

        append_history_mock = AsyncMock()
        with patch("movies.views.aclassify_and_parse", _mock_classify_and_parse("follow_up", semantic_query="расскажи подробнее")), \
             patch("movies.views._append_history", append_history_mock), \
             patch("movies.views.astream_conversational", _one_token_stream):

            view = ChatView.as_view()
            response = await view(request)

            assert response.status_code == 200
            body = b""
            async for chunk in response.streaming_content:
                body += chunk if isinstance(chunk, bytes) else chunk.encode()

            text = body.decode()
            assert "event: session" in text
            assert "event: done" in text
            assert "event: movies" not in text

        assert append_history_mock.await_count == 2
        user_call, assistant_call = append_history_mock.await_args_list
        assert user_call.args[1] == "user"
        assert user_call.args[2] == "расскажи подробнее"
        assert assistant_call.args[1] == "assistant"
        assert assistant_call.kwargs == {"count_turn": False}

    @pytest.mark.asyncio
    async def test_general_chat_returns_text_only(self):
        factory = AsyncRequestFactory()
        request = factory.post(
            "/api/chat/",
            data=json.dumps({"message": "привет"}),
            content_type="application/json",
        )

        append_history_mock = AsyncMock()
        with patch("movies.views.aclassify_and_parse", _mock_classify_and_parse("general_chat", semantic_query="привет")), \
             patch("movies.views._append_history", append_history_mock), \
             patch("movies.views.astream_conversational", _one_token_stream):

            view = ChatView.as_view()
            response = await view(request)

            assert response.status_code == 200
            body = b""
            async for chunk in response.streaming_content:
                body += chunk if isinstance(chunk, bytes) else chunk.encode()

            text = body.decode()
            assert "event: session" in text
            assert "event: movies" not in text

        assert append_history_mock.await_count == 2

    @pytest.mark.asyncio
    async def test_assistant_turn_store_failure_still_sends_done(self):
        factory = AsyncRequestFactory()
        request = factory.post(
            "/api/chat/",
            data=json.dumps({"message": "привет"}),
            content_type="application/json",
        )

        # First call (user turn) succeeds, second call (assistant turn) fails.
        append_history_mock = AsyncMock(side_effect=[None, RuntimeError("db down")])
        with patch("movies.views.aclassify_and_parse", _mock_classify_and_parse("general_chat", semantic_query="привет")), \
             patch("movies.views._append_history", append_history_mock), \
             patch("movies.views.astream_conversational", _one_token_stream):

            view = ChatView.as_view()
            response = await view(request)

            assert response.status_code == 200
            body = b""
            async for chunk in response.streaming_content:
                body += chunk if isinstance(chunk, bytes) else chunk.encode()

            text = body.decode()
            assert "event: done" in text

        assert append_history_mock.await_count == 2

    @pytest.mark.asyncio
    async def test_clarify_asks_with_clarify_prompt_and_no_movies(self):
        factory = AsyncRequestFactory()
        request = factory.post(
            "/api/chat/",
            data=json.dumps({"message": "movies"}),
            content_type="application/json",
        )

        conversational = MagicMock(side_effect=_one_token_stream)
        generate_mock = AsyncMock()
        with patch("movies.views.aclassify_and_parse", _mock_classify_and_parse("clarify", semantic_query="movies")), \
             patch("movies.views._append_history", AsyncMock()), \
             patch("movies.views._generate_and_score", generate_mock), \
             patch("movies.views.astream_conversational", conversational):

            view = ChatView.as_view()
            response = await view(request)

            assert response.status_code == 200
            body = b""
            async for chunk in response.streaming_content:
                body += chunk if isinstance(chunk, bytes) else chunk.encode()

            text = body.decode()
            assert "event: movies" not in text
            assert "event: done" in text

        generate_mock.assert_not_called()
        assert conversational.call_args.kwargs["system_prompt"] == CLARIFY_SYSTEM_PROMPT


@pytest.mark.django_db
class TestChatSSEEmptyReply:
    @staticmethod
    async def _read(response):
        body = b""
        async for chunk in response.streaming_content:
            body += chunk if isinstance(chunk, bytes) else chunk.encode()
        return body.decode()

    @pytest.mark.asyncio
    async def test_empty_conversational_stream_stores_no_assistant_entry(self):
        from movies.models import ChatSession

        session = await ChatSession.objects.acreate()
        request = AsyncRequestFactory().post(
            "/api/chat/",
            data=json.dumps({"message": "привет", "session_id": str(session.session_id)}),
            content_type="application/json",
            headers={"X-Session-Token": session.session_token},
        )

        with patch("movies.views.aclassify_and_parse", _mock_classify_and_parse("general_chat", semantic_query="привет")), \
             patch("movies.views.astream_conversational", _empty_stream):
            response = await ChatView.as_view()(request)
            text = await self._read(response)

        assert "event: done" in text
        await session.arefresh_from_db()
        assert [e["role"] for e in session.history] == ["user"]

    @pytest.mark.asyncio
    async def test_empty_explanation_stream_still_records_shown_movies(self):
        from movies.models import ChatSession

        session = await ChatSession.objects.acreate()
        request = AsyncRequestFactory().post(
            "/api/chat/",
            data=json.dumps({"message": "хочу комедию", "session_id": str(session.session_id)}),
            content_type="application/json",
            headers={"X-Session-Token": session.session_token},
        )

        with patch("movies.views.aclassify_and_parse", _mock_classify_and_parse("new_search", semantic_query="хочу комедию")), \
             patch("movies.views.encode_query", return_value=[0.1] * 1024), \
             patch("movies.views._generate_and_score", _mock_generate_and_score()), \
             patch("movies.views.astream_explanation_parts", _empty_parts_stream):
            response = await ChatView.as_view()(request)
            text = await self._read(response)

        assert "event: error" in text
        assert "event: done" in text
        await session.arefresh_from_db()
        # The cards were shown, so the turn is kept (empty content + movies) for
        # the LLM context to know what was recommended.
        assistant_entries = [e for e in session.history if e["role"] == "assistant"]
        assert assistant_entries == [{"role": "assistant", "content": "", "movies": ["Test Film"]}]

    @pytest.mark.asyncio
    async def test_whitespace_only_conversational_reply_is_not_stored(self):
        async def _blank(*args, **kwargs):
            yield "  "
            yield "\n"

        append_history_mock = AsyncMock()
        request = AsyncRequestFactory().post(
            "/api/chat/",
            data=json.dumps({"message": "привет"}),
            content_type="application/json",
        )
        with patch("movies.views.aclassify_and_parse", _mock_classify_and_parse("general_chat", semantic_query="привет")), \
             patch("movies.views._append_history", append_history_mock), \
             patch("movies.views.astream_conversational", _blank):
            response = await ChatView.as_view()(request)
            text = await self._read(response)

        assert "event: done" in text
        append_history_mock.assert_awaited_once()
        assert append_history_mock.await_args.args[1] == "user"


@pytest.mark.django_db
class TestChatSSENoMatch:
    """The relevance gate (reranking.min_score) can leave no results."""

    @staticmethod
    async def _read(response):
        body = b""
        async for chunk in response.streaming_content:
            body += chunk if isinstance(chunk, bytes) else chunk.encode()
        return body.decode()

    @staticmethod
    def _request(session, message="фильм про несуществующее"):
        return AsyncRequestFactory().post(
            "/api/chat/",
            data=json.dumps({"message": message, "session_id": str(session.session_id)}),
            content_type="application/json",
            headers={"X-Session-Token": session.session_token},
        )

    @staticmethod
    async def _no_results(query_embedding, intent, query_text="", exclude_ids=None):
        return [], []

    @pytest.mark.asyncio
    async def test_empty_gated_result_emits_no_match_and_fixed_message_without_llm(self):
        from movies.models import ChatSession
        from movies.views import NO_MATCH_MESSAGE

        session = await ChatSession.objects.acreate()
        explain = MagicMock(side_effect=AssertionError("explanation must not be called"))
        with patch("movies.views.aclassify_and_parse", _mock_classify_and_parse("new_search", semantic_query="x")), \
             patch("movies.views.encode_query", return_value=[0.1] * 1024), \
             patch("movies.views._generate_and_score", self._no_results), \
             patch("movies.views.astream_explanation_parts", explain):
            response = await ChatView.as_view()(self._request(session))
            events = _parse_sse(await self._read(response))

        assert [name for name, _ in events] == ["movies", "token", "done"]
        movies_event = dict(events)["movies"]
        assert movies_event["movies"] == []
        assert movies_event["no_match"] is True
        assert movies_event["session_id"] == str(session.session_id)
        assert dict(events)["token"] == {"text": NO_MATCH_MESSAGE}
        explain.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_result_saves_session_and_stores_text_without_movies(self):
        from movies.models import ChatSession
        from movies.views import NO_MATCH_MESSAGE

        session = await ChatSession.objects.acreate()
        with patch("movies.views.aclassify_and_parse", _mock_classify_and_parse("new_search", semantic_query="x")), \
             patch("movies.views.encode_query", return_value=[0.1] * 1024), \
             patch("movies.views._generate_and_score", self._no_results):
            response = await ChatView.as_view()(self._request(session))
            await self._read(response)

        await session.arefresh_from_db()
        assert session.history == [
            {"role": "user", "content": "фильм про несуществующее"},
            {"role": "assistant", "content": NO_MATCH_MESSAGE},
        ]
        assert session.turn_count == 1

    @pytest.mark.asyncio
    async def test_fewer_than_top_n_movies_take_the_normal_explanation_path(self):
        from movies.models import ChatSession

        session = await ChatSession.objects.acreate()
        two = _two_movies()
        with patch("movies.views.aclassify_and_parse", _mock_classify_and_parse("new_search", semantic_query="x")), \
             patch("movies.views.encode_query", return_value=[0.1] * 1024), \
             patch("movies.views._generate_and_score", _mock_generate_and_score(two)), \
             patch("movies.views.astream_explanation_parts",
                   _mock_explanation_parts((None, "Две находки."), (0, "Первый."), (1, "Второй."))):
            response = await ChatView.as_view()(self._request(session))
            events = _parse_sse(await self._read(response))

        movies_event = dict(events)["movies"]
        assert [m["serial_name"] for m in movies_event["movies"]] == ["Film 1", "Film 2"]
        assert movies_event["no_match"] is False
        assert [name for name, _ in events if name == "blurb"] == ["blurb", "blurb"]


@pytest.mark.django_db
class TestChatSSERefinement:
    @pytest.mark.asyncio
    async def test_refinement_returns_new_movies(self):
        factory = AsyncRequestFactory()
        request = factory.post(
            "/api/chat/",
            data=json.dumps({"message": "а повеселее?"}),
            content_type="application/json",
        )

        with patch("movies.views.aclassify_and_parse", _mock_classify_and_parse("refinement", semantic_query="а повеселее?")), \
             patch("movies.views.encode_query", return_value=[0.1] * 1024), \
             patch("movies.views._generate_and_score", _mock_generate_and_score()), \
             patch("movies.views._save_session", AsyncMock()), \
             patch("movies.views._append_history", AsyncMock()), \
             patch("movies.views.astream_explanation_parts", _empty_parts_stream):

            view = ChatView.as_view()
            response = await view(request)

            assert response.status_code == 200
            body = b""
            async for chunk in response.streaming_content:
                body += chunk if isinstance(chunk, bytes) else chunk.encode()

            text = body.decode()
            assert "event: movies" in text
            assert "event: done" in text


def test_chat_session_has_no_preference_vector():
    from movies.models import ChatSession

    assert "preference_vector" not in {f.name for f in ChatSession._meta.get_fields()}


class TestUnsupportedConstraintsAndRelaxation:
    """No DB: the session and both history writers are patched, so these run on
    a host without Postgres. `history` is what the session held before the turn."""

    @staticmethod
    def _session(history=()):
        session = MagicMock()
        session.session_id = "00000000-0000-0000-0000-000000000001"
        session.session_token = "token"
        session.history = list(history)
        return session

    async def _run(self, intent, history=(), generate=None, explain=None):
        append_history = AsyncMock()
        generate = generate or _mock_generate_and_score()
        explain = explain or _mock_explanation_parts((None, "Here you go."))
        request = AsyncRequestFactory().post(
            "/api/chat/", data=json.dumps({"message": "msg"}), content_type="application/json",
        )

        async def _classify(message, history=None):
            return intent

        with patch("movies.views._get_or_create_session", AsyncMock(return_value=self._session(history))), \
             patch("movies.views.aclassify_and_parse", _classify), \
             patch("movies.views.encode_query", return_value=[0.1] * 1024), \
             patch("movies.views._generate_and_score", generate), \
             patch("movies.views._save_session", AsyncMock()), \
             patch("movies.views._append_history", append_history), \
             patch("movies.views.astream_explanation_parts", explain):
            response = await ChatView.as_view()(request)
            body = b""
            async for chunk in response.streaming_content:
                body += chunk if isinstance(chunk, bytes) else chunk.encode()
        return _parse_sse(body.decode()), append_history

    @staticmethod
    def _recording_generate(calls, movies=None):
        inner = _mock_generate_and_score(movies)

        async def _gen(query_embedding, intent, query_text="", exclude_ids=None):
            calls.append(exclude_ids)
            return await inner(query_embedding, intent, query_text)
        return _gen

    @pytest.mark.asyncio
    async def test_unsupported_constraint_asks_instead_of_searching(self):
        calls = []
        explain = MagicMock(side_effect=AssertionError("no LLM for this reply"))
        intent = MessageIntent(
            category="new_search", semantic_query="1950s movies", min_release_year=1950,
            max_release_year=1959, unsupported_constraints=["with Keanu Reeves"],
        )
        events, append_history = await self._run(
            intent, generate=self._recording_generate(calls), explain=explain,
        )

        assert calls == []
        explain.assert_not_called()
        assert [name for name, _ in events] == ["movies", "token", "done"]
        assert dict(events)["movies"]["movies"] == []
        text = dict(events)["token"]["text"]
        assert "with Keanu Reeves" in text
        assert "search without" in text
        call = append_history.await_args
        assert call.args[1:3] == ("assistant", text)
        assert call.kwargs["pending_relaxation"] == {"dropped": ["with Keanu Reeves"]}

    @pytest.mark.asyncio
    async def test_after_the_offer_the_dropped_constraint_no_longer_blocks_the_search(self):
        calls = []
        history = [
            {"role": "user", "content": "1950s movies with Keanu Reeves"},
            {"role": "assistant", "content": "I can't check...",
             "pending_relaxation": {"dropped": ["with Keanu Reeves"]}},
        ]
        intent = MessageIntent(
            category="new_search", semantic_query="1950s movies",
            unsupported_constraints=["With Keanu Reeves "],
        )
        events, _ = await self._run(intent, history=history, generate=self._recording_generate(calls))

        assert len(calls) == 1
        assert [m["serial_name"] for m in dict(events)["movies"]["movies"]] == ["Test Film"]

    @pytest.mark.asyncio
    async def test_the_offer_survives_an_intervening_chat_turn(self):
        calls = []
        history = [
            {"role": "user", "content": "silent documentaries"},
            {"role": "assistant", "content": "I can't check...", "pending_relaxation": {"dropped": ["silent films"]}},
            {"role": "user", "content": "why?"},
            {"role": "assistant", "content": "Because the catalog has no such data."},
        ]
        intent = MessageIntent(category="new_search", semantic_query="documentaries", unsupported_constraints=["silent"])
        events, _ = await self._run(intent, history=history, generate=self._recording_generate(calls))

        assert len(calls) == 1
        assert [m["serial_name"] for m in dict(events)["movies"]["movies"]] == ["Test Film"]

    @staticmethod
    def _turn_lines(caplog):
        return [r.getMessage() for r in caplog.records if r.getMessage().startswith("search_turn ")]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("movies, results", [(None, 1), ([], 0)])
    async def test_a_search_turn_logs_one_diagnostics_line(self, caplog, movies, results):
        caplog.set_level(logging.INFO, logger="movies.views")
        generate = _mock_generate_and_score(movies) if movies is None else self._empty_generate()
        intent = MessageIntent(
            category="refinement", semantic_query="x", standalone_query="slow westerns",
            unsupported_constraints=["silent"],
        )
        history = [{"role": "assistant", "pending_relaxation": {"dropped": ["silent"]}}]
        await self._run(intent, history=history, generate=generate)

        assert self._turn_lines(caplog) == [
            f"search_turn category=refinement unsupported_constraints=['silent'] "
            f"standalone_query='slow westerns' results={results}"
        ]

    @pytest.mark.asyncio
    async def test_the_unsupported_offer_logs_one_diagnostics_line_with_no_results(self, caplog):
        caplog.set_level(logging.INFO, logger="movies.views")
        intent = MessageIntent(
            category="new_search", standalone_query="silent documentaries", unsupported_constraints=["silent"],
        )
        await self._run(intent)

        assert self._turn_lines(caplog) == [
            "search_turn category=new_search unsupported_constraints=['silent'] "
            "standalone_query='silent documentaries' results=0"
        ]

    @staticmethod
    def _empty_generate():
        async def _gen(query_embedding, intent, query_text="", exclude_ids=None):
            return [], []
        return _gen

    def test_unchecked_constraints_match_loosely_in_either_direction(self):
        history = [{"role": "assistant", "pending_relaxation": {"dropped": ["Silent", "with Keanu Reeves"]}}]

        assert _unchecked_constraints(["silent films", "keanu"], history) == []
        assert _unchecked_constraints(["directed by Kubrick"], history) == ["directed by Kubrick"]

    def test_unchecked_constraints_read_the_whole_window_but_not_beyond_it(self):
        offer = {"role": "assistant", "pending_relaxation": {"dropped": ["silent"]}}
        filler = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
        window = settings.SESSION_HISTORY_MAX_TURNS

        assert _unchecked_constraints(["silent"], [offer] + filler * (window - 1)) == []
        assert _unchecked_constraints(["silent"], [offer] + filler * window) == ["silent"]

    def test_unchecked_constraints_ignore_an_empty_offer(self):
        history = [{"role": "assistant", "pending_relaxation": {"dropped": [" "]}}]

        assert _unchecked_constraints(["silent"], history) == ["silent"]

    @pytest.mark.asyncio
    async def test_a_new_unsupported_constraint_still_asks_after_an_offer(self):
        calls = []
        history = [{"role": "assistant", "content": "...", "pending_relaxation": {"dropped": ["silent"]}}]
        intent = MessageIntent(category="refinement", unsupported_constraints=["directed by Kubrick"])
        events, _ = await self._run(intent, history=history, generate=self._recording_generate(calls))

        assert calls == []
        assert "directed by Kubrick" in dict(events)["token"]["text"]

    @pytest.mark.asyncio
    async def test_shown_movies_are_stored_as_tmdb_ids(self):
        events, append_history = await self._run(
            MessageIntent(category="new_search", semantic_query="x"),
            generate=_mock_generate_and_score(_two_movies()),
        )
        assert append_history.await_args.kwargs["movie_ids"] == [1, 2]

    @pytest.mark.asyncio
    async def test_refinement_excludes_every_movie_already_shown(self):
        calls = []
        history = [
            {"role": "assistant", "content": "a", "movies": ["A", "B"], "movie_ids": [10, 11]},
            {"role": "user", "content": "funnier"},
            {"role": "assistant", "content": "b", "movies": ["C"], "movie_ids": [12]},
        ]
        await self._run(
            MessageIntent(category="refinement", semantic_query="x"),
            history=history, generate=self._recording_generate(calls),
        )
        assert calls == [{10, 11, 12}]

    @pytest.mark.asyncio
    async def test_refinement_exclusion_stays_inside_the_history_window(self, settings):
        settings.SESSION_HISTORY_MAX_TURNS = 1
        calls = []
        history = [
            {"role": "assistant", "content": "a", "movies": ["A"], "movie_ids": [10]},
            {"role": "user", "content": "funnier"},
            {"role": "assistant", "content": "b", "movies": ["C"], "movie_ids": [12]},
        ]
        await self._run(
            MessageIntent(category="refinement", semantic_query="x"),
            history=history, generate=self._recording_generate(calls),
        )
        assert calls == [{12}]

    @pytest.mark.asyncio
    async def test_new_search_does_not_exclude_shown_movies(self):
        calls = []
        history = [{"role": "assistant", "content": "a", "movies": ["A"], "movie_ids": [10]}]
        await self._run(
            MessageIntent(category="new_search", semantic_query="x"),
            history=history, generate=self._recording_generate(calls),
        )
        assert calls == [None]

    @pytest.mark.asyncio
    async def test_empty_result_lists_the_active_filters(self):
        async def _no_results(query_embedding, intent, query_text="", exclude_ids=None):
            return [], []

        intent = MessageIntent(
            category="new_search", semantic_query="x", min_release_year=1950, max_release_year=1959,
            max_runtime=20, min_vote_average=9.0,
        )
        events, _ = await self._run(intent, generate=_no_results)
        assert dict(events)["token"]["text"] == _no_match_message(
            intent.model_dump(exclude=set(NON_FILTER_FIELDS)),
        )
        assert dict(events)["movies"]["no_match"] is True


    # Fixed "Only N movie(s)" line: fewer than top_n results under hard filters.
    FILTERED = dict(category="new_search", semantic_query="thrillers", max_release_year=1959)

    @pytest.mark.asyncio
    async def test_line_precedes_the_llm_intro_and_is_stored(self):
        intent = MessageIntent(**self.FILTERED)
        events, append_history = await self._run(
            intent, explain=_mock_explanation_parts((None, "Here you go.\n"), (0, "Desc.")),
        )

        line = "Only 1 movie in the catalog matches: released 1959 or earlier."
        tokens = [d["text"] for name, d in events if name == "token"]
        assert tokens[0].strip() == line
        assert "".join(tokens).endswith("Here you go.\n")
        stored = append_history.call_args.args[2]
        assert stored.startswith(line + "\n")
        assert stored.count(line) == 1
        assert "Here you go." in stored and "[[1]]\nDesc." in stored

    @pytest.mark.asyncio
    async def test_singular_and_plural_agree_with_the_count(self):
        intent = MessageIntent(**self.FILTERED)
        with patch.object(settings, "TOP_N", 3):
            events, _ = await self._run(
                intent, generate=_mock_generate_and_score(_two_movies()),
                explain=_mock_explanation_parts((None, "Hi.")),
            )
        assert [d["text"] for name, d in events if name == "token"][0].strip() == (
            "Only 2 movies in the catalog match: released 1959 or earlier."
        )
        events, _ = await self._run(intent, explain=_mock_explanation_parts((None, "Hi.")))
        assert [d["text"] for name, d in events if name == "token"][0].strip() == (
            "Only 1 movie in the catalog matches: released 1959 or earlier."
        )

    @pytest.mark.asyncio
    async def test_no_intro_final_when_the_model_streams_nothing(self):
        intent = MessageIntent(**self.FILTERED)
        events, _ = await self._run(intent, explain=_mock_explanation_parts())
        names = [name for name, _ in events]
        assert "error" in names and "intro_final" not in names

    @pytest.mark.asyncio
    async def test_intro_final_keeps_the_line(self):
        intent = MessageIntent(**self.FILTERED)
        events, append_history = await self._run(
            intent, explain=_mock_explanation_parts((None, "Opening.\nstray second line")),
        )
        final = dict(events)["intro_final"]["text"]
        assert final.startswith("Only 1 movie in the catalog matches:")
        assert final.endswith("Opening.")
        assert append_history.call_args.args[2] == final

    @pytest.mark.asyncio
    async def test_no_line_when_top_n_movies_come_back(self):
        intent = MessageIntent(**self.FILTERED)
        with patch.object(settings, "TOP_N", 2):
            events, append_history = await self._run(
                intent, generate=_mock_generate_and_score(_two_movies()),
            )
        assert "Only" not in "".join(d["text"] for name, d in events if name == "token")
        assert "Only" not in append_history.call_args.args[2]

    @pytest.mark.asyncio
    async def test_no_line_on_a_refinement_that_excluded_shown_movies(self):
        # The shortfall may come from dropping the movies already shown, not
        # from the catalog running out, so the server must not claim a count.
        intent = MessageIntent(**{**self.FILTERED, "category": "refinement"})
        history = [{"role": "assistant", "content": "a", "movies": ["A"], "movie_ids": [10]}]
        events, append_history = await self._run(intent, history=history)
        assert "Only" not in "".join(d["text"] for name, d in events if name == "token")
        assert "Only" not in append_history.call_args.args[2]

    @pytest.mark.asyncio
    async def test_no_line_when_reference_films_may_have_been_excluded(self):
        intent = MessageIntent(**self.FILTERED, reference_films=["Alien"])
        events, append_history = await self._run(intent)
        assert "Only" not in "".join(d["text"] for name, d in events if name == "token")
        assert "Only" not in append_history.call_args.args[2]

    @pytest.mark.asyncio
    async def test_no_line_without_hard_filters(self):
        intent = MessageIntent(category="new_search", semantic_query="thrillers", genres=["Thriller"])
        events, append_history = await self._run(intent)
        assert "Only" not in "".join(d["text"] for name, d in events if name == "token")
        assert "Only" not in append_history.call_args.args[2]


class TestDescribeFilters:
    def test_empty_when_only_ranking_fields_are_set(self):
        assert _describe_filters({"genres": ["Drama"], "mood": "sad"}) == ""

    def test_joins_active_filters(self):
        assert _describe_filters({"negations": ["Horror"], "max_runtime": 20}) == "no Horror, under 20 min"


class TestNoMatchMessage:
    def test_lists_each_active_filter(self):
        text = _no_match_message({
            "genres": ["Documentary"], "negations": ["Horror"], "country_inclusions": ["France"],
            "country_exclusions": ["USA"], "min_vote_average": 9.0, "min_release_year": 1950,
            "max_release_year": 1959, "min_runtime": None, "max_runtime": 20,
            "original_languages": ["fr"], "mood": "happy", "themes": ["war"], "reference_films": [],
        })
        assert text == (
            "Nothing in the catalog matches all of: no Horror, from France, not from USA, "
            "rating ≥ 9.0, released 1950–1959, under 20 min, language: fr. "
            "Which one should I loosen?"
        )

    def test_one_sided_year_and_runtime_bounds(self):
        text = _no_match_message({"min_release_year": 2015, "min_runtime": 150})
        assert "released 2015 or later" in text
        assert "at least 150 min" in text

    def test_no_hard_filters_falls_back_to_the_generic_message(self):
        assert _no_match_message({"genres": ["Drama"], "mood": "sad"}) == NO_MATCH_MESSAGE
