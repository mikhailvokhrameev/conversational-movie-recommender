import asyncio
import json
import time
from unittest.mock import patch, AsyncMock

import pytest
from django.conf import settings
from django.test import AsyncRequestFactory

from core.session_manager import update_preference_vector
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


@pytest.mark.django_db
class TestSaveSession:
    @pytest.mark.asyncio
    async def test_refinement_uses_higher_alpha_than_new_search(self):
        from movies.models import ChatSession

        existing_vector = [0.1] * 1024
        new_embedding = [0.9] + [0.0] * 1023

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


class TestSaveSessionConcurrency:
    @pytest.mark.django_db(transaction=True)
    @pytest.mark.asyncio
    async def test_concurrent_saves_both_apply_under_row_lock(self, monkeypatch):
        """Two concurrent saves must compose (v0 -> e1 -> e2 in some order), not
        overwrite each other. The slow wrapper below widens the read-modify-write
        window so the test fails deterministically if the row lock is removed:
        without it both threads read v0 and one update is lost."""
        from movies.models import ChatSession

        real_update = update_preference_vector

        def _slow_update(*args, **kwargs):
            result = real_update(*args, **kwargs)
            time.sleep(0.2)
            return result

        monkeypatch.setattr("movies.views.update_preference_vector", _slow_update)

        v0 = [0.1] * 1024
        e1 = [0.9] + [0.0] * 1023
        e2 = [0.0, 0.9] + [0.0] * 1022
        session = await ChatSession.objects.acreate(preference_vector=v0)

        await asyncio.gather(
            _save_session(session, "q1", {}, e1, [], category="new_search"),
            _save_session(session, "q2", {}, e2, [], category="new_search"),
        )
        await session.arefresh_from_db()

        order_12 = real_update(real_update(v0, e1), e2)
        order_21 = real_update(real_update(v0, e2), e1)
        final = list(session.preference_vector)
        assert final == pytest.approx(order_12) or final == pytest.approx(order_21)
        assert {e["content"] for e in session.history if e["role"] == "user"} == {"q1", "q2"}
        assert len(session.history) == 2
        assert session.turn_count == 2


@pytest.mark.django_db
class TestPreferenceVectorUsedForScoring:
    @staticmethod
    async def _run_turn(category, existing_vector, query_embedding):
        from core.ollama_client import MessageIntent
        from movies.models import ChatSession

        session = await ChatSession.objects.acreate(preference_vector=existing_vector)
        intent = MessageIntent(category=category, semantic_query="q")

        async def _classify(message, history=None):
            return intent

        generate = AsyncMock(return_value=([], []))
        factory = AsyncRequestFactory()
        request = factory.post(
            "/api/chat/",
            data=json.dumps({"message": "q", "session_id": str(session.session_id)}),
            content_type="application/json",
            headers={"X-Session-Token": session.session_token},
        )

        async def _no_parts(query, movies, history=None):
            return
            yield

        with patch("movies.views.aclassify_and_parse", _classify), \
             patch("movies.views.encode_query", return_value=query_embedding), \
             patch("movies.views._generate_and_score", generate), \
             patch("movies.views._append_history", AsyncMock()), \
             patch("movies.views.astream_explanation_parts", _no_parts):
            response = await ChatView.as_view()(request)
            async for _ in response.streaming_content:
                pass

        await session.arefresh_from_db()
        return generate.await_args.args[2], session

    @pytest.mark.asyncio
    async def test_refinement_scores_with_updated_vector_and_saves_the_same(self):
        existing = [0.1] * 1024
        query = [0.9] + [0.0] * 1023

        scored, session = await self._run_turn("refinement", existing, query)

        expected = update_preference_vector(existing, query, alpha=settings.SESSION_ALPHA_REFINEMENT)
        assert list(scored) == pytest.approx(expected)
        assert list(session.preference_vector) == pytest.approx(list(scored))

    @pytest.mark.asyncio
    async def test_new_search_scores_with_pre_turn_vector(self):
        existing = [0.1] * 1024
        query = [0.9] + [0.0] * 1023

        scored, session = await self._run_turn("new_search", existing, query)

        assert list(scored) == pytest.approx(existing)
        assert list(session.preference_vector) == pytest.approx(
            update_preference_vector(existing, query, alpha=settings.SESSION_ALPHA)
        )


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

        await _save_session(
            session, "q", {}, [0.1] * 1024,
            [{"serial_name": "Test Film"}], category="new_search",
        )
        await session.arefresh_from_db()

        assert session.history == [{"role": "user", "content": "q"}]
