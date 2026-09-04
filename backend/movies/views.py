import asyncio
import json
import logging
import secrets
from datetime import datetime

from asgiref.sync import sync_to_async
from django.db import transaction
from django.http import JsonResponse, StreamingHttpResponse
from django.views import View
from rest_framework.response import Response
from rest_framework.views import APIView

from core.candidate_generation import generate_candidates
from core.embedding_service import encode_query
from core.ollama_client import (
    aclassify_message, aparse_intent,
    astream_conversational, astream_explanation,
)
from core.scoring import mmr_diversify, score_candidates
from core.session_manager import track_explicit_preferences, update_preference_vector
from .models import ChatSession, Movie

logger = logging.getLogger(__name__)

TOP_N = 5


class HealthView(APIView):
    def get(self, request):
        return Response({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "catalog_size": Movie.objects.count(),
        })


def _serialize_movie(movie: dict) -> dict:
    return {
        "id": movie["id"],
        "serial_name": movie["serial_name"],
        "genres": movie["genres"],
        "content_type": movie["content_type"],
        "country": movie["country"],
        "actors": movie["actors"],
        "director": movie["director"],
        "age_rating": float(movie["age_rating"]) if movie["age_rating"] is not None else None,
        "release_date": movie["release_date"],
        "description": movie["description"],
        "url": movie["url"],
        "score": round(float(movie["total"]), 4),
    }


async def _get_or_create_session(session_id: str | None) -> ChatSession:
    if session_id:
        try:
            session = await ChatSession.objects.aget(session_id=session_id)
            if not session.is_expired():
                return session
        except ChatSession.DoesNotExist:
            pass
    return await ChatSession.objects.acreate()


@sync_to_async(thread_sensitive=False)
def _save_session(session: ChatSession, query: str, intent: dict,
                  query_embedding: list[float], movies: list[dict]):
    with transaction.atomic():
        fresh = ChatSession.objects.select_for_update().get(pk=session.pk)
        fresh.preference_vector = update_preference_vector(
            [float(x) for x in fresh.preference_vector] if fresh.preference_vector is not None else None,
            query_embedding,
        )
        fresh.preferences = track_explicit_preferences(fresh.preferences, intent)
        fresh.history = list(fresh.history) + [{
            "role": "user",
            "content": query,
            "movies": [m["serial_name"] for m in movies],
        }]
        fresh.turn_count += 1
        fresh.save()


@sync_to_async(thread_sensitive=False)
def _append_history(session: ChatSession, role: str, content: str):
    with transaction.atomic():
        fresh = ChatSession.objects.select_for_update().get(pk=session.pk)
        fresh.history = list(fresh.history) + [{"role": role, "content": content}]
        fresh.turn_count += 1
        fresh.save()


@sync_to_async(thread_sensitive=False)
def _generate_and_score(query_embedding, intent, session_vector):
    candidates = generate_candidates(query_embedding, intent)
    scored = score_candidates(candidates, query_embedding, intent, session_vector)
    return mmr_diversify(scored, top_n=TOP_N)


def _last_movies_context(session: ChatSession) -> str:
    for entry in reversed(session.history):
        movies = entry.get("movies")
        if movies:
            return "Последние рекомендованные фильмы: " + ", ".join(movies) + "."
    return ""


class ChatView(View):
    async def post(self, request):
        try:
            body = json.loads(request.body)
        except (json.JSONDecodeError, ValueError):
            return JsonResponse({"error": "invalid JSON"}, status=400)

        message = body.get("message", "").strip()
        if not message:
            return JsonResponse({"error": "message is required"}, status=400)
        if len(message) > 2000:
            return JsonResponse({"error": "message too long (max 2000 chars)"}, status=400)

        try:
            session_id = body.get("session_id")
            session = await _get_or_create_session(session_id)
            category = await aclassify_message(message)
        except Exception:
            logger.exception("ChatView classification error")
            return JsonResponse({"error": "internal error"}, status=500)

        if category in ("follow_up", "general_chat"):
            return await self._handle_conversational(session, message, category)
        elif category == "refinement":
            return await self._handle_refinement(session, message)
        else:
            return await self._handle_new_search(session, message)

    async def _handle_new_search(self, session, message):
        try:
            intent, query_embedding = await asyncio.gather(
                aparse_intent(message),
                sync_to_async(encode_query)(message),
            )
            session_vector = [float(x) for x in session.preference_vector] if session.preference_vector is not None else None
            top_movies = await _generate_and_score(query_embedding, intent, session_vector)
            serialized = [_serialize_movie(m) for m in top_movies]
            await _save_session(session, message, intent, query_embedding, top_movies)
        except Exception:
            logger.exception("ChatView new_search error")
            return JsonResponse({"error": "internal error"}, status=500)

        movies_for_llm = [
            {"serial_name": m["serial_name"], "genres": m["genres"], "description": m["description"]}
            for m in top_movies
        ]

        async def event_stream():
            yield _sse_event("movies", {
                "session_id": str(session.session_id),
                "session_token": session.session_token,
                "movies": serialized,
                "intent": intent,
            })
            has_tokens = False
            async for token in astream_explanation(message, movies_for_llm):
                has_tokens = True
                yield _sse_event("token", {"text": token})
            if not has_tokens:
                yield _sse_event("error", {"message": "explanation generation failed"})
            yield _sse_event("done", {})

        return StreamingHttpResponse(
            event_stream(),
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def _handle_conversational(self, session, message, category):
        context = ""
        if category == "follow_up":
            context = _last_movies_context(session)

        await _append_history(session, "user", message)

        async def event_stream():
            yield _sse_event("session", {"session_id": str(session.session_id), "session_token": session.session_token})
            async for token in astream_conversational(message, context):
                yield _sse_event("token", {"text": token})
            yield _sse_event("done", {})

        return StreamingHttpResponse(
            event_stream(),
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def _handle_refinement(self, session, message):
        try:
            intent, query_embedding = await asyncio.gather(
                aparse_intent(message),
                sync_to_async(encode_query)(message),
            )
            session_vector = [float(x) for x in session.preference_vector] if session.preference_vector is not None else None
            top_movies = await _generate_and_score(query_embedding, intent, session_vector)
            serialized = [_serialize_movie(m) for m in top_movies]
            await _save_session(session, message, intent, query_embedding, top_movies)
        except Exception:
            logger.exception("ChatView refinement error")
            return JsonResponse({"error": "internal error"}, status=500)

        movies_for_llm = [
            {"serial_name": m["serial_name"], "genres": m["genres"], "description": m["description"]}
            for m in top_movies
        ]

        async def event_stream():
            yield _sse_event("movies", {
                "session_id": str(session.session_id),
                "session_token": session.session_token,
                "movies": serialized,
                "intent": intent,
            })
            has_tokens = False
            async for token in astream_explanation(message, movies_for_llm):
                has_tokens = True
                yield _sse_event("token", {"text": token})
            if not has_tokens:
                yield _sse_event("error", {"message": "explanation generation failed"})
            yield _sse_event("done", {})

        return StreamingHttpResponse(
            event_stream(),
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class SessionHistoryView(View):
    async def get(self, request, session_id):
        token = request.headers.get("X-Session-Token", "")
        if not token:
            return JsonResponse({"error": "session token required"}, status=401)
        try:
            session = await ChatSession.objects.aget(session_id=session_id)
        except ChatSession.DoesNotExist:
            return JsonResponse({"error": "session not found"}, status=404)
        if not secrets.compare_digest(token, session.session_token):
            return JsonResponse({"error": "invalid session token"}, status=403)
        return JsonResponse({
            "session_id": str(session.session_id),
            "history": session.history,
            "turn_count": session.turn_count,
            "preferences": session.preferences,
        })
