import json
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
from django.conf import settings
from django.test import AsyncRequestFactory

from core.session_manager import update_preference_vector
from movies.views import ChatView, SessionHistoryView, _save_session, _serialize_movie


class TestSerializeMovie:
    def test_serializes_all_fields(self):
        movie = {
            "id": 1,
            "serial_name": "Test Film",
            "genres": ["Драмы"],
            "content_type": "Фильм",
            "country": ["Россия"],
            "actors": ["Actor"],
            "director": "Director",
            "age_rating": 16.0,
            "release_date": "2024-01-01",
            "description": "A test film",
            "url": "https://okko.tv/test",
            "total": 0.87654,
        }
        result = _serialize_movie(movie)
        assert result["score"] == 0.8765
        assert result["serial_name"] == "Test Film"
        assert "total" not in result
        assert "embedding" not in result

    def test_null_age_rating(self):
        movie = {
            "id": 1, "serial_name": "T", "genres": [], "content_type": "",
            "country": [], "actors": [], "director": "", "age_rating": None,
            "release_date": None, "description": "", "url": "", "total": 0.5,
        }
        result = _serialize_movie(movie)
        assert result["age_rating"] is None


@pytest.mark.django_db
class TestChatViewValidation:
    @pytest.mark.asyncio
    async def test_empty_message_returns_400(self):
        factory = AsyncRequestFactory()
        request = factory.post(
            "/api/chat/",
            data=json.dumps({"message": ""}),
            content_type="application/json",
        )
        view = ChatView.as_view()
        response = await view(request)
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_missing_message_returns_400(self):
        factory = AsyncRequestFactory()
        request = factory.post(
            "/api/chat/",
            data=json.dumps({}),
            content_type="application/json",
        )
        view = ChatView.as_view()
        response = await view(request)
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_message_too_long_returns_400(self):
        factory = AsyncRequestFactory()
        request = factory.post(
            "/api/chat/",
            data=json.dumps({"message": "x" * 2001}),
            content_type="application/json",
        )
        view = ChatView.as_view()
        response = await view(request)
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self):
        factory = AsyncRequestFactory()
        request = factory.post(
            "/api/chat/",
            data="not json",
            content_type="application/json",
        )
        view = ChatView.as_view()
        response = await view(request)
        assert response.status_code == 400


@pytest.mark.django_db
class TestSaveSession:
    @pytest.mark.asyncio
    async def test_refinement_uses_higher_alpha_than_new_search(self):
        from movies.models import ChatSession

        existing_vector = [0.1] * 768
        new_embedding = [0.9] + [0.0] * 767

        session_new_search = await ChatSession.objects.acreate(preference_vector=existing_vector)
        await _save_session(session_new_search, "q", {}, new_embedding, [], category="new_search")
        await session_new_search.arefresh_from_db()

        session_refinement = await ChatSession.objects.acreate(preference_vector=existing_vector)
        await _save_session(session_refinement, "q", {}, new_embedding, [], category="refinement")
        await session_refinement.arefresh_from_db()

        expected_new_search = update_preference_vector(existing_vector, new_embedding, alpha=settings.SESSION_ALPHA)
        expected_refinement = update_preference_vector(existing_vector, new_embedding, alpha=settings.SESSION_ALPHA_REFINEMENT)

        assert list(session_new_search.preference_vector) == pytest.approx(expected_new_search)
        assert list(session_refinement.preference_vector) == pytest.approx(expected_refinement)
        assert list(session_new_search.preference_vector) != pytest.approx(list(session_refinement.preference_vector))


@pytest.mark.django_db
class TestSessionHistoryView:
    @pytest.mark.asyncio
    async def test_missing_token_returns_401(self):
        from movies.models import ChatSession
        session = await ChatSession.objects.acreate()

        factory = AsyncRequestFactory()
        request = factory.get(f"/api/sessions/{session.session_id}/")
        view = SessionHistoryView.as_view()
        response = await view(request, session_id=session.session_id)
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_wrong_token_returns_403(self):
        from movies.models import ChatSession
        session = await ChatSession.objects.acreate()

        factory = AsyncRequestFactory()
        request = factory.get(
            f"/api/sessions/{session.session_id}/",
            headers={"x-session-token": "wrong-token"},
        )
        view = SessionHistoryView.as_view()
        response = await view(request, session_id=session.session_id)
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_valid_token_returns_200(self):
        from movies.models import ChatSession
        session = await ChatSession.objects.acreate()

        factory = AsyncRequestFactory()
        request = factory.get(
            f"/api/sessions/{session.session_id}/",
            headers={"x-session-token": session.session_token},
        )
        view = SessionHistoryView.as_view()
        response = await view(request, session_id=session.session_id)
        assert response.status_code == 200
        body = json.loads(response.content)
        assert body["session_id"] == str(session.session_id)

    @pytest.mark.asyncio
    async def test_nonexistent_session_returns_404(self):
        import uuid
        factory = AsyncRequestFactory()
        request = factory.get(
            f"/api/sessions/{uuid.uuid4()}/",
            headers={"x-session-token": "any"},
        )
        view = SessionHistoryView.as_view()
        response = await view(request, session_id=uuid.uuid4())
        assert response.status_code == 404
