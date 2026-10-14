import inspect
import json
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
from django.test import AsyncRequestFactory

from core.ollama_client import MessageIntent, aclassify_and_parse, astream_explanation_parts
from movies.views import ChatView, _generate_and_score


def _mock_classify_and_parse(category, semantic_query="", **filters):
    intent = MessageIntent(category=category, semantic_query=semantic_query, **filters)

    async def _classify(message, history=None):
        return intent
    return _classify


def _mock_encode(embedding=None):
    return embedding or [0.1] * 1024


def _mock_generate_and_score(movies=None):
    async def _gen(query_embedding, intent, session_vector, query_text=""):
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

        # Stored as paragraphs (intro, then each description in slot order),
        # not the raw concatenation of chunks.
        append_history_mock.assert_awaited_once()
        call = append_history_mock.await_args
        assert call.args[1] == "assistant"
        assert call.args[2] == "Вот подборка комедий.\n\nПро первый фильм.\n\nПро второй фильм."
        assert call.kwargs["movies"] == ["Film 1", "Film 2"]

    @pytest.mark.asyncio
    async def test_assistant_turn_joins_intro_and_descriptions_with_blank_lines(self):
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
        assert append_history_mock.await_args.args[2] == "Intro.\n\nDesc one."


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
