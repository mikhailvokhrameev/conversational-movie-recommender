import asyncio
import json
import time

import pytest
from django.conf import settings
from django.test import AsyncRequestFactory

from core.session_manager import track_explicit_preferences
from movies.views import (
    ChatView, SessionHistoryView, _get_or_create_session, _save_session, _serialize_movie,
)


class TestSerializeMovie:
    def test_serializes_all_fields(self):
        movie = {
            "id": 1,
            "tmdb_id": 42,
            "serial_name": "Test Film",
            "original_title": "Test Film",
            "genres": ["Drama"],
            "country": ["US"],
            "release_date": "2024-01-01",
            "description": "A test film",
            "runtime": 120,
            "vote_average": 7.5,
            "poster_path": "/abc123.jpg",
            "total": 0.87654,
        }
        result = _serialize_movie(movie)
        assert result["score"] == 0.8765
        assert result["serial_name"] == "Test Film"
        assert result["poster_url"] == f"{settings.TMDB_POSTER_BASE_URL}{settings.TMDB_POSTER_SIZE}/abc123.jpg"
        assert "total" not in result
        assert "embedding" not in result

    def test_null_poster_path_yields_null_poster_url(self):
        movie = {
            "id": 1, "tmdb_id": 1, "serial_name": "T", "original_title": "T",
            "genres": [], "country": [], "release_date": None, "description": "",
            "runtime": None, "vote_average": 0.0, "poster_path": None, "total": 0.5,
        }
        result = _serialize_movie(movie)
        assert result["poster_url"] is None


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
class TestGetOrCreateSession:
    @pytest.mark.asyncio
    async def test_no_session_id_creates_new_session(self):
        session = await _get_or_create_session(None, "")
        assert session.pk is not None

    @pytest.mark.asyncio
    async def test_valid_token_resumes_existing_session(self):
        from movies.models import ChatSession
        existing = await ChatSession.objects.acreate()

        session = await _get_or_create_session(str(existing.session_id), existing.session_token)

        assert session.pk == existing.pk

    @pytest.mark.asyncio
    async def test_missing_token_does_not_resume_existing_session(self):
        from movies.models import ChatSession
        existing = await ChatSession.objects.acreate()

        session = await _get_or_create_session(str(existing.session_id), "")

        assert session.pk != existing.pk

    @pytest.mark.asyncio
    async def test_wrong_token_does_not_resume_existing_session(self):
        from movies.models import ChatSession
        existing = await ChatSession.objects.acreate()

        session = await _get_or_create_session(str(existing.session_id), "wrong-token")

        assert session.pk != existing.pk


class TestSaveSessionConcurrency:
    @pytest.mark.django_db(transaction=True)
    @pytest.mark.asyncio
    async def test_concurrent_saves_both_apply_under_row_lock(self, monkeypatch):
        """Two concurrent saves must compose, not overwrite each other. The slow
        wrapper widens the read-modify-write window so the test fails
        deterministically if the row lock is removed: without it both threads
        read the same preferences and one update is lost."""
        from movies.models import ChatSession

        real_track = track_explicit_preferences

        def _slow_track(*args, **kwargs):
            result = real_track(*args, **kwargs)
            time.sleep(0.2)
            return result

        monkeypatch.setattr("movies.views.track_explicit_preferences", _slow_track)

        session = await ChatSession.objects.acreate()

        await asyncio.gather(
            _save_session(session, "q1", {"genres": ["Drama"]}),
            _save_session(session, "q2", {"genres": ["Comedy"]}),
        )
        await session.arefresh_from_db()

        assert set(session.preferences["liked_genres"]) == {"Drama", "Comedy"}
        assert {e["content"] for e in session.history if e["role"] == "user"} == {"q1", "q2"}
        assert len(session.history) == 2
        assert session.turn_count == 2


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


@pytest.mark.django_db
class TestAppendHistory:
    @pytest.mark.asyncio
    async def test_appends_entry_with_role_and_content(self):
        from movies.models import ChatSession
        from movies.views import _append_history
        session = await ChatSession.objects.acreate()

        await _append_history(session, "user", "hello")
        await session.arefresh_from_db()

        assert session.history == [{"role": "user", "content": "hello"}]

    @pytest.mark.asyncio
    async def test_assistant_entry_includes_movies_when_given(self):
        from movies.models import ChatSession
        from movies.views import _append_history
        session = await ChatSession.objects.acreate()

        await _append_history(session, "assistant", "Here you go", movies=["Film A", "Film B"])
        await session.arefresh_from_db()

        assert session.history == [{
            "role": "assistant", "content": "Here you go",
            "movies": ["Film A", "Film B"],
        }]

    @pytest.mark.asyncio
    async def test_assistant_entry_includes_movie_facts_when_given(self):
        from movies.models import ChatSession
        from movies.views import _append_history
        session = await ChatSession.objects.acreate()
        facts = [{"title": "Film A", "year": 1999, "genres": ["Drama"], "runtime": 120,
                  "description": "About A."}]

        await _append_history(
            session, "assistant", "Here you go", movies=["Film A"], movie_facts=facts,
        )
        await session.arefresh_from_db()

        assert session.history == [{
            "role": "assistant", "content": "Here you go",
            "movies": ["Film A"], "movie_facts": facts,
        }]

    @pytest.mark.asyncio
    async def test_count_turn_false_does_not_increment_turn_count(self):
        from movies.models import ChatSession
        from movies.views import _append_history
        session = await ChatSession.objects.acreate()

        await _append_history(session, "user", "hello")
        await session.arefresh_from_db()
        assert session.turn_count == 1

        await _append_history(session, "assistant", "hi back", count_turn=False)
        await session.arefresh_from_db()
        assert session.turn_count == 1
        assert len(session.history) == 2


@pytest.mark.django_db
class TestSaveSessionHistoryEntry:
    @pytest.mark.asyncio
    async def test_user_entry_has_no_movies_key(self):
        from movies.models import ChatSession
        session = await ChatSession.objects.acreate()

        await _save_session(session, "q", {})
        await session.arefresh_from_db()

        assert session.history == [{"role": "user", "content": "q"}]
