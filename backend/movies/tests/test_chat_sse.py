import inspect
import json
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
from django.test import AsyncRequestFactory

from core.ollama_client import MessageIntent, astream_explanation_parts
from movies.views import ChatView, _generate_and_score


def _mock_classify_and_parse(category, semantic_query="", **filters):
    intent = MessageIntent(category=category, semantic_query=semantic_query, **filters)

    async def _classify(msg):
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


async def _empty_parts_stream(query, movies):
    """Stand-in for astream_explanation_parts: same (slot, text) contract, no output."""
    return
    yield


def _mock_explanation_parts(*parts):
    """Stand-in for astream_explanation_parts yielding the given (slot, text) pairs."""
    async def _stream(query, movies):
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

        with patch("movies.views.aclassify_and_parse", _mock_classify_and_parse("new_search", semantic_query="хочу комедию")), \
             patch("movies.views.encode_query", return_value=[0.1] * 1024), \
             patch("movies.views._generate_and_score", _mock_generate_and_score()), \
             patch("movies.views._save_session", AsyncMock()), \
             patch("movies.views.astream_explanation_parts", _empty_parts_stream):

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

        with patch("movies.views.aclassify_and_parse", _mock_classify_and_parse("new_search", semantic_query="хочу комедию")), \
             patch("movies.views.encode_query", return_value=[0.1] * 1024), \
             patch("movies.views._generate_and_score", _mock_generate_and_score(_two_movies())), \
             patch("movies.views._save_session", AsyncMock()), \
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

        with patch("movies.views.aclassify_and_parse", _mock_classify_and_parse("follow_up", semantic_query="расскажи подробнее")), \
             patch("movies.views._append_history", AsyncMock()), \
             patch("movies.views.astream_conversational", _empty_stream):

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

    @pytest.mark.asyncio
    async def test_general_chat_returns_text_only(self):
        factory = AsyncRequestFactory()
        request = factory.post(
            "/api/chat/",
            data=json.dumps({"message": "привет"}),
            content_type="application/json",
        )

        with patch("movies.views.aclassify_and_parse", _mock_classify_and_parse("general_chat", semantic_query="привет")), \
             patch("movies.views._append_history", AsyncMock()), \
             patch("movies.views.astream_conversational", _empty_stream):

            view = ChatView.as_view()
            response = await view(request)

            assert response.status_code == 200
            body = b""
            async for chunk in response.streaming_content:
                body += chunk if isinstance(chunk, bytes) else chunk.encode()

            text = body.decode()
            assert "event: session" in text
            assert "event: movies" not in text


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
