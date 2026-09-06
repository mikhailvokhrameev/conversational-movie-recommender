import json
import logging
import secrets
import time
from datetime import datetime

from asgiref.sync import sync_to_async
from django.conf import settings
from django.db import transaction
from django.http import JsonResponse, StreamingHttpResponse
from django.views import View
from rest_framework.response import Response
from rest_framework.views import APIView

from core.candidate_generation import generate_candidates
from core.embedding_service import encode_query
from core.ollama_client import (
    aclassify_and_parse, check_explanation_titles,
    astream_conversational, astream_explanation,
)
from core.reranking import rerank_candidates
from core.scoring import mmr_diversify, score_candidates
from core.session_manager import track_explicit_preferences, update_preference_vector
from .models import ChatSession, Movie

logger = logging.getLogger(__name__)


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
                  query_embedding: list[float], movies: list[dict], category: str):
    with transaction.atomic():
        fresh = ChatSession.objects.select_for_update().get(pk=session.pk)
        alpha = settings.SESSION_ALPHA_REFINEMENT if category == "refinement" else None
        fresh.preference_vector = update_preference_vector(
            [float(x) for x in fresh.preference_vector] if fresh.preference_vector is not None else None,
            query_embedding,
            alpha=alpha,
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
def _generate_and_score(query_embedding, intent, session_vector, query_text=""):
    """Returns (top_movies, candidate_pool): candidate_pool is the full reranked
    pool before the top-n cut, kept around cheaply for the hallucination check
    in astream_explanation's caller -- no extra retrieval work, no DB round trip."""
    candidates = generate_candidates(query_embedding, intent, query_text=query_text)
    scored = score_candidates(candidates, query_embedding, intent, session_vector)
    reranked = rerank_candidates(query_text, scored)
    top_movies = mmr_diversify(reranked, top_n=settings.TOP_N)
    return top_movies, reranked


def _log_turn_latency(category, **stage_ms):
    logger.info(
        "turn_latency category=%s %s",
        category,
        " ".join(f"{stage}={ms:.1f}ms" for stage, ms in stage_ms.items() if ms is not None),
    )


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
            t0 = time.perf_counter()
            intent = await aclassify_and_parse(message)
            classify_parse_ms = (time.perf_counter() - t0) * 1000
        except Exception:
            logger.exception("ChatView classification error")
            return JsonResponse({"error": "internal error"}, status=500)

        if intent.category in ("follow_up", "general_chat"):
            return await self._handle_conversational(session, message, intent.category, classify_parse_ms)
        else:
            return await self._handle_search_like(session, message, intent, classify_parse_ms)

    async def _handle_search_like(self, session, message, intent, classify_parse_ms):
        """Handles both `new_search` and `refinement`: identical pipeline, differing
        only in the EMA alpha `_save_session` picks based on `intent.category`."""
        try:
            t_embed = time.perf_counter()
            query_embedding = await sync_to_async(encode_query)(intent.semantic_query)
            embed_ms = (time.perf_counter() - t_embed) * 1000

            intent_dict = intent.model_dump(exclude={"category", "semantic_query"})
            session_vector = [float(x) for x in session.preference_vector] if session.preference_vector is not None else None

            t_score = time.perf_counter()
            top_movies, candidate_pool = await _generate_and_score(query_embedding, intent_dict, session_vector, query_text=message)
            score_ms = (time.perf_counter() - t_score) * 1000

            serialized = [_serialize_movie(m) for m in top_movies]
            await _save_session(session, message, intent_dict, query_embedding, top_movies, category=intent.category)
        except Exception:
            logger.exception(f"ChatView {intent.category} error")
            return JsonResponse({"error": "internal error"}, status=500)

        movies_for_llm = [
            {"serial_name": m["serial_name"], "genres": m["genres"], "description": m["description"]}
            for m in top_movies
        ]
        top_titles = [m["serial_name"] for m in top_movies]
        candidate_titles = [m["serial_name"] for m in candidate_pool]

        async def event_stream():
            yield _sse_event("movies", {
                "session_id": str(session.session_id),
                "session_token": session.session_token,
                "movies": serialized,
                "intent": intent_dict,
            })
            has_tokens = False
            explanation_parts = []
            t_explain = time.perf_counter()
            async for token in astream_explanation(message, movies_for_llm):
                has_tokens = True
                explanation_parts.append(token)
                yield _sse_event("token", {"text": token})
            explain_ms = (time.perf_counter() - t_explain) * 1000
            if not has_tokens:
                yield _sse_event("error", {"message": "explanation generation failed"})

            suspicious = check_explanation_titles("".join(explanation_parts), top_titles, candidate_titles)
            if suspicious:
                logger.warning(f"Explanation may reference non-recommended titles {suspicious} for query={message!r}")
            _log_turn_latency(
                intent.category, classify_parse_ms=classify_parse_ms,
                embed_ms=embed_ms, score_ms=score_ms, explain_ms=explain_ms,
            )
            yield _sse_event("done", {})

        return StreamingHttpResponse(
            event_stream(),
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def _handle_conversational(self, session, message, category, classify_parse_ms):
        context = ""
        if category == "follow_up":
            context = _last_movies_context(session)

        await _append_history(session, "user", message)

        async def event_stream():
            yield _sse_event("session", {"session_id": str(session.session_id), "session_token": session.session_token})
            t_explain = time.perf_counter()
            async for token in astream_conversational(message, context):
                yield _sse_event("token", {"text": token})
            explain_ms = (time.perf_counter() - t_explain) * 1000
            _log_turn_latency(category, classify_parse_ms=classify_parse_ms, explain_ms=explain_ms)
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
