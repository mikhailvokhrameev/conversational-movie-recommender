import inspect
import json
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
from django.test import AsyncRequestFactory

from core.ollama_client import MessageIntent
from movies.views import ChatView, _generate_and_score


def _mock_classify_and_parse(category, semantic_query="", **filters):
    intent = MessageIntent(category=category, semantic_query=semantic_query, **filters)

    async def _classify(msg):
        return intent
    return _classify


def _mock_encode(embedding=None):
    return embedding or [0.1] * 768


def _mock_generate_and_score(movies=None):
    async def _gen(query_embedding, intent, session_vector, query_text=""):
        result = movies or [
            {
                "id": 1, "serial_name": "Test Film", "genres": ["Драмы"],
                "content_type": "Фильм", "country": ["Россия"], "actors": [],
                "director": "Director", "age_rating": 16.0, "release_date": "2024-01-01",
                "description": "A test", "url": "https://okko.tv/test",
                "embedding": [0.1] * 768, "total": 0.85, "semantic": 0.4,
                "metadata": 0.3, "session": 0.15,
            }
        ]
        return result, result
    return _gen


async def _empty_stream(*args, **kwargs):
    return
    yield


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
             patch("movies.views.encode_query", return_value=[0.1] * 768), \
             patch("movies.views._generate_and_score", _mock_generate_and_score()), \
             patch("movies.views._save_session", AsyncMock()), \
             patch("movies.views.astream_explanation", _empty_stream):

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
        encode_mock = MagicMock(return_value=[0.1] * 768)

        with patch("movies.views.aclassify_and_parse", _mock_classify_and_parse("new_search", semantic_query="весёлая комедия")), \
             patch("movies.views.encode_query", encode_mock), \
             patch("movies.views._generate_and_score", _mock_generate_and_score()), \
             patch("movies.views._save_session", AsyncMock()), \
             patch("movies.views.astream_explanation", _empty_stream):

            view = ChatView.as_view()
            response = await view(request)
            async for _ in response.streaming_content:
                pass

            encode_mock.assert_called_once_with("весёлая комедия")


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
             patch("movies.views.encode_query", return_value=[0.1] * 768), \
             patch("movies.views._generate_and_score", _mock_generate_and_score()), \
             patch("movies.views._save_session", AsyncMock()), \
             patch("movies.views.astream_explanation", _empty_stream):

            view = ChatView.as_view()
            response = await view(request)

            assert response.status_code == 200
            body = b""
            async for chunk in response.streaming_content:
                body += chunk if isinstance(chunk, bytes) else chunk.encode()

            text = body.decode()
            assert "event: movies" in text
            assert "event: done" in text
