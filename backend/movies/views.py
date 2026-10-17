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

from core.embedding_service import encode_query
from core.ollama_client import (
    aclassify_and_parse, check_explanation_titles,
    astream_conversational, astream_explanation_parts,
)
from core.pipeline import generate_and_score
from core.session_manager import (
    build_movie_facts, track_explicit_preferences,
)
from .models import ChatSession, Movie

logger = logging.getLogger(__name__)

# Reply when the relevance gate (reranking.min_score) leaves no results. A
# fixed user-facing string, not a tunable: no LLM call, so a "nothing found"
# answer cannot be dressed up by the model into recommendations from thin air.
NO_MATCH_MESSAGE = (
    "Nothing in the catalog matched that well. "
    "Try rephrasing your request or loosening some of the constraints."
)


class HealthView(APIView):
    def get(self, request):
        return Response({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "catalog_size": Movie.objects.count(),
        })


def _serialize_movie(movie: dict) -> dict:
    poster_path = movie["poster_path"]
    return {
        "id": movie["id"],
        "tmdb_id": movie["tmdb_id"],
        "serial_name": movie["serial_name"],
        "original_title": movie["original_title"],
        "genres": movie["genres"],
        "country": movie["country"],
        "release_date": movie["release_date"],
        "description": movie["description"],
        "runtime": movie["runtime"],
        "vote_average": movie["vote_average"],
        "poster_url": (
            f"{settings.TMDB_POSTER_BASE_URL}{settings.TMDB_POSTER_SIZE}{poster_path}"
            if poster_path else None
        ),
        "score": round(float(movie["total"]), 4),
    }


async def _get_or_create_session(session_id: str | None, token: str) -> ChatSession:
    if session_id and token:
        try:
            session = await ChatSession.objects.aget(session_id=session_id)
            if not session.is_expired() and secrets.compare_digest(token, session.session_token):
                return session
        except ChatSession.DoesNotExist:
            pass
    return await ChatSession.objects.acreate()


@sync_to_async(thread_sensitive=False)
def _save_session(session: ChatSession, query: str, intent: dict, movies: list[dict]):
    """Read-modify-write under a row lock, so concurrent turns compose instead
    of overwriting each other's preferences, history and turn count."""
    with transaction.atomic():
        fresh = ChatSession.objects.select_for_update().get(pk=session.pk)
        fresh.preferences = track_explicit_preferences(fresh.preferences, intent)
        fresh.history = list(fresh.history) + [{"role": "user", "content": query}]
        fresh.turn_count += 1
        fresh.save()


@sync_to_async(thread_sensitive=False)
def _append_history(session: ChatSession, role: str, content: str,
                    movies: list[str] | None = None, count_turn: bool = True,
                    movie_facts: list[dict] | None = None):
    with transaction.atomic():
        fresh = ChatSession.objects.select_for_update().get(pk=session.pk)
        entry = {"role": role, "content": content}
        if movies:
            entry["movies"] = movies
        if movie_facts:
            entry["movie_facts"] = movie_facts
        fresh.history = list(fresh.history) + [entry]
        if count_turn:
            fresh.turn_count += 1
        fresh.save()


@sync_to_async(thread_sensitive=False)
def _generate_and_score(query_embedding, intent, query_text=""):
    """Async wrapper over `core.pipeline.generate_and_score` -- the same function
    `evaluate_scoring` runs, so what is evaluated is what ships."""
    return generate_and_score(query_embedding, intent, query_text=query_text)


def _log_turn_latency(category, **stage_ms):
    logger.info(
        "turn_latency category=%s %s",
        category,
        " ".join(f"{stage}={ms:.1f}ms" for stage, ms in stage_ms.items() if ms is not None),
    )


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
            token = request.headers.get("X-Session-Token", "")
            session = await _get_or_create_session(session_id, token)
            t0 = time.perf_counter()
            intent = await aclassify_and_parse(message, session.history)
            classify_parse_ms = (time.perf_counter() - t0) * 1000
        except Exception:
            logger.exception("ChatView classification error")
            return JsonResponse({"error": "internal error"}, status=500)

        if intent.category in ("follow_up", "general_chat"):
            return await self._handle_conversational(session, message, intent.category, classify_parse_ms)
        else:
            return await self._handle_search_like(session, message, intent, classify_parse_ms)

    async def _handle_search_like(self, session, message, intent, classify_parse_ms):
        """Handles both `new_search` and `refinement`: one identical pipeline."""
        # Captured before _save_session/_append_history run below -- both write
        # via a separately-fetched row and never mutate this in-memory `session`,
        # so this deliberately reflects history *before* this turn: the current
        # message/movies are added as the final turn by astream_explanation_parts
        # itself, and including them here too would duplicate them.
        history = session.history
        try:
            t_embed = time.perf_counter()
            query_embedding = await sync_to_async(encode_query)(intent.semantic_query)
            embed_ms = (time.perf_counter() - t_embed) * 1000

            intent_dict = intent.model_dump(exclude={"category", "semantic_query"})

            t_score = time.perf_counter()
            top_movies, candidate_pool = await _generate_and_score(query_embedding, intent_dict, query_text=message)
            score_ms = (time.perf_counter() - t_score) * 1000

            serialized = [_serialize_movie(m) for m in top_movies]
            await _save_session(session, message, intent_dict, top_movies)
        except Exception:
            logger.exception(f"ChatView {intent.category} error")
            return JsonResponse({"error": "internal error"}, status=500)

        if not top_movies:
            return self._no_match_response(
                session, intent, intent_dict, classify_parse_ms, embed_ms, score_ms,
            )

        movies_for_llm = [
            {"serial_name": m["serial_name"], "genres": m["genres"], "description": m["description"]}
            for m in top_movies
        ]
        top_titles = [m["serial_name"] for m in top_movies]
        movie_facts = build_movie_facts(top_movies, settings.SESSION_HISTORY_MOVIE_DESCRIPTION_CHARS)
        candidate_titles = [m["serial_name"] for m in candidate_pool]

        async def event_stream():
            yield _sse_event("movies", {
                "session_id": str(session.session_id),
                "session_token": session.session_token,
                "movies": serialized,
                "no_match": False,
                "intent": intent_dict,
            })
            has_tokens = False
            explanation_parts = []
            intro_parts = []
            slot_parts = {}
            t_explain = time.perf_counter()
            # The opening line keeps the plain "token" wire format so it streams
            # live under the existing caret; each per-movie description goes out
            # as a "blurb" addressed to the card it belongs to.
            async for slot, text in astream_explanation_parts(message, movies_for_llm, history):
                has_tokens = True
                explanation_parts.append(text)
                if slot is None:
                    intro_parts.append(text)
                    yield _sse_event("token", {"text": text})
                else:
                    slot_parts.setdefault(slot, []).append(text)
                    yield _sse_event("blurb", {"index": slot, "text": text})
            explain_ms = (time.perf_counter() - t_explain) * 1000
            if not has_tokens:
                yield _sse_event("error", {"message": "explanation generation failed"})

            # Intro and each per-movie description are joined with blank lines
            # (in slot order) so the stored turn reads as separate paragraphs
            # rather than blurbs glued together.
            assistant_blocks = ["".join(intro_parts)] + [
                "".join(slot_parts[slot]) for slot in sorted(slot_parts)
            ]
            assistant_text = "\n\n".join(b for b in assistant_blocks if b)
            # Skip storing only when there is nothing to keep: with an empty
            # explanation the movies were still shown, so store the entry with
            # empty content and `movies` -- the history formatter turns it into
            # "(Recommended: ...)" (or, for the latest set, a facts block built
            # from `movie_facts`) so later turns know what was recommended.
            # Otherwise the user turn would stay without a reply -- acceptable,
            # Ollama takes consecutive user messages; the 2*N window shifts slightly.
            if assistant_text.strip() or top_titles:
                try:
                    await _append_history(
                        session, "assistant", assistant_text,
                        movies=top_titles, count_turn=False, movie_facts=movie_facts,
                    )
                except Exception:
                    logger.exception("failed to store assistant turn")

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

    def _no_match_response(self, session, intent, intent_dict, classify_parse_ms, embed_ms, score_ms):
        """The relevance gate left nothing: say so instead of explaining top-5.

        The movies event carries `no_match: true` and an empty list, the fixed
        message goes out as a normal token reply, and the stored assistant turn
        has no `movies`/`movie_facts` -- nothing was shown, so a later follow-up
        must not be grounded in movies the user never saw.
        """
        async def event_stream():
            yield _sse_event("movies", {
                "session_id": str(session.session_id),
                "session_token": session.session_token,
                "movies": [],
                "no_match": True,
                "intent": intent_dict,
            })
            yield _sse_event("token", {"text": NO_MATCH_MESSAGE})
            try:
                await _append_history(session, "assistant", NO_MATCH_MESSAGE, count_turn=False)
            except Exception:
                logger.exception("failed to store assistant turn")
            _log_turn_latency(
                intent.category, classify_parse_ms=classify_parse_ms,
                embed_ms=embed_ms, score_ms=score_ms,
            )
            yield _sse_event("done", {})

        return StreamingHttpResponse(
            event_stream(),
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def _handle_conversational(self, session, message, category, classify_parse_ms):
        # See the matching comment in _handle_search_like: captured before the
        # user-turn _append_history call below, which writes via a separately-
        # fetched row and doesn't mutate this in-memory `session`.
        history = session.history

        await _append_history(session, "user", message)

        async def event_stream():
            yield _sse_event("session", {"session_id": str(session.session_id), "session_token": session.session_token})
            t_explain = time.perf_counter()
            reply_parts = []
            async for token in astream_conversational(message, history):
                reply_parts.append(token)
                yield _sse_event("token", {"text": token})
            explain_ms = (time.perf_counter() - t_explain) * 1000
            reply = "".join(reply_parts)
            # Same guard as in _handle_search_like: no empty assistant entry; the
            # orphaned user turn stays in history (Ollama accepts consecutive user
            # messages; the 2*N window shifts slightly).
            if reply.strip():
                try:
                    await _append_history(session, "assistant", reply, count_turn=False)
                except Exception:
                    logger.exception("failed to store assistant turn")
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
