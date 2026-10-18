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
    CLARIFY_SYSTEM_PROMPT, CONVERSATIONAL_SYSTEM_PROMPT, NON_FILTER_FIELDS,
    aclassify_and_parse, check_explanation_titles,
    astream_conversational, astream_explanation_parts,
)
from core.pipeline import generate_and_score
from core.session_manager import (
    build_movie_facts, track_explicit_preferences,
)
from .models import ChatSession, Movie

logger = logging.getLogger(__name__)

# Fixed user-facing replies, not tunables: no LLM call, so a "nothing found"
# or "can't check that" answer cannot be dressed up by the model into
# recommendations from thin air.
NO_MATCH_MESSAGE = (
    "Nothing in the catalog matched that well. "
    "Try rephrasing your request or loosening some of the constraints."
)
UNSUPPORTED_MESSAGE = (
    "I can't check {constraints} -- the catalog has no data on {unchecked}. "
    "Want me to search without that?"
)


def _describe_filters(intent_dict: dict) -> str:
    """The hard filters that were active, comma-separated; empty with none.
    Genres, mood and themes only rank, they never empty the result, so they
    are not listed."""
    parts = [f"no {g}" for g in intent_dict.get("negations") or []]
    parts += [f"from {c}" for c in intent_dict.get("country_inclusions") or []]
    parts += [f"not from {c}" for c in intent_dict.get("country_exclusions") or []]
    if intent_dict.get("min_vote_average") is not None:
        parts.append(f"rating ≥ {intent_dict['min_vote_average']}")
    min_year, max_year = intent_dict.get("min_release_year"), intent_dict.get("max_release_year")
    if min_year is not None and max_year is not None:
        parts.append(f"released {min_year}–{max_year}")
    elif min_year is not None:
        parts.append(f"released {min_year} or later")
    elif max_year is not None:
        parts.append(f"released {max_year} or earlier")
    if intent_dict.get("min_runtime") is not None:
        parts.append(f"at least {intent_dict['min_runtime']} min")
    if intent_dict.get("max_runtime") is not None:
        parts.append(f"under {intent_dict['max_runtime']} min")
    if intent_dict.get("original_languages"):
        parts.append(f"language: {', '.join(intent_dict['original_languages'])}")
    return ", ".join(parts)


def _no_match_message(intent_dict: dict) -> str:
    """Name the hard filters that were active, so the user knows what to
    loosen; with no hard filter the cause was the relevance gate."""
    filters = _describe_filters(intent_dict)
    if not filters:
        return NO_MATCH_MESSAGE
    return f"Nothing in the catalog matches all of: {filters}. Which one should I loosen?"


def _history_window(history: list[dict]) -> list[dict]:
    """The last session.history_max_turns pairs. Stored history is not capped,
    so the window keeps what is replayed bounded and in step with what the
    model still sees of the conversation."""
    return history[-(settings.SESSION_HISTORY_MAX_TURNS * 2):]


def _unchecked_constraints(constraints: list[str], history: list[dict]) -> list[str]:
    """The intent's unsupported constraints minus those an assistant turn in the
    history window already offered to drop: a "yes" or "okay" to that offer
    searches even after a chat turn in between, and even if the classifier
    words the constraint differently ("silent films" vs "silent")."""
    dropped = {
        c.strip().casefold() for entry in _history_window(history) if entry.get("role") == "assistant"
        for c in (entry.get("pending_relaxation") or {}).get("dropped") or []
    }
    dropped.discard("")

    def _offered(constraint: str) -> bool:
        key = constraint.strip().casefold()
        return any(d in key or key in d for d in dropped)

    return [c for c in constraints if not _offered(c)]


def _shown_movie_ids(history: list[dict]) -> set[int]:
    """tmdb_ids of the movies shown within the replayed history window, for a
    refinement to skip."""
    return {
        movie_id for entry in _history_window(history) if entry.get("role") == "assistant"
        for movie_id in entry.get("movie_ids") or []
    }


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


def _assistant_turn_text(intro: str, slot_parts: dict[int, list[str]]) -> str:
    """The stored assistant turn in the shape the explanation prompt asks for:
    the opening line, then each description under its 1-based [[n]] marker.
    Replaying the markers in history keeps the model producing them on later
    turns; a history without them taught it to drop the format."""
    blocks = [intro.strip()] if intro.strip() else []
    for slot in sorted(slot_parts):
        text = "".join(slot_parts[slot]).strip()
        if text:
            blocks.append(f"[[{slot + 1}]]\n{text}")
    return "\n".join(blocks)


def _first_line(text: str) -> str:
    """The opening line of a reply that ignored the marker format."""
    return text.strip().split("\n", 1)[0].strip()


@sync_to_async(thread_sensitive=False)
def _append_history(session: ChatSession, role: str, content: str,
                    movies: list[str] | None = None, count_turn: bool = True,
                    movie_facts: list[dict] | None = None, movie_ids: list[int] | None = None,
                    pending_relaxation: dict | None = None):
    with transaction.atomic():
        fresh = ChatSession.objects.select_for_update().get(pk=session.pk)
        entry = {"role": role, "content": content}
        if movies:
            entry["movies"] = movies
        if movie_facts:
            entry["movie_facts"] = movie_facts
        if movie_ids:
            entry["movie_ids"] = movie_ids
        if pending_relaxation:
            entry["pending_relaxation"] = pending_relaxation
        fresh.history = list(fresh.history) + [entry]
        if count_turn:
            fresh.turn_count += 1
        fresh.save()


@sync_to_async(thread_sensitive=False)
def _generate_and_score(query_embedding, intent, query_text="", exclude_ids=None):
    """Async wrapper over `core.pipeline.generate_and_score` -- the same function
    `evaluate_scoring` runs, so what is evaluated is what ships."""
    return generate_and_score(query_embedding, intent, query_text=query_text, exclude_ids=exclude_ids)


def _log_turn_latency(category, **stage_ms):
    logger.info(
        "turn_latency category=%s %s",
        category,
        " ".join(f"{stage}={ms:.1f}ms" for stage, ms in stage_ms.items() if ms is not None),
    )


def _log_search_turn(intent, results):
    """One line per search turn, whichever way it ends (listed, empty, or the offer)."""
    logger.info(
        "search_turn category=%s unsupported_constraints=%s standalone_query=%r results=%d",
        intent.category, intent.unsupported_constraints, intent.standalone_query, results,
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

        if intent.category in ("follow_up", "general_chat", "clarify"):
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
        intent_dict = intent.model_dump(exclude=set(NON_FILTER_FIELDS))
        unchecked = _unchecked_constraints(intent.unsupported_constraints, history)
        if unchecked:
            _log_search_turn(intent, 0)
            return await self._unsupported_response(
                session, message, intent, intent_dict, unchecked, classify_parse_ms,
            )
        # A refinement must not hand back what the user has already seen; a new
        # search may legitimately land on an earlier movie again.
        exclude_ids = _shown_movie_ids(history) if intent.category == "refinement" else None
        try:
            t_embed = time.perf_counter()
            query_embedding = await sync_to_async(encode_query)(intent.semantic_query)
            embed_ms = (time.perf_counter() - t_embed) * 1000

            t_score = time.perf_counter()
            top_movies, candidate_pool = await _generate_and_score(
                query_embedding, intent_dict, query_text=intent.standalone_query, exclude_ids=exclude_ids,
            )
            score_ms = (time.perf_counter() - t_score) * 1000

            serialized = [_serialize_movie(m) for m in top_movies]
            await _save_session(session, message, intent_dict, top_movies)
        except Exception:
            logger.exception(f"ChatView {intent.category} error")
            return JsonResponse({"error": "internal error"}, status=500)

        _log_search_turn(intent, len(top_movies))
        if not top_movies:
            return self._no_match_response(
                session, intent, intent_dict, classify_parse_ms, embed_ms, score_ms,
            )

        movies_for_llm = [
            {"serial_name": m["serial_name"], "genres": m["genres"], "description": m["description"]}
            for m in top_movies
        ]
        top_titles = [m["serial_name"] for m in top_movies]
        top_ids = [m["tmdb_id"] for m in top_movies]
        movie_facts = build_movie_facts(top_movies, settings.SESSION_HISTORY_MOVIE_DESCRIPTION_CHARS)
        candidate_titles = [m["serial_name"] for m in candidate_pool]
        # Fewer than top_n under hard filters: say so before the model's intro,
        # in fixed words, so the intro cannot pass the short list off as a full one.
        short_filters = _describe_filters(intent_dict) if len(top_movies) < settings.TOP_N else ""
        count = len(top_movies)
        short_line = (
            f"Only {count} movie in the catalog matches: {short_filters}.\n" if count == 1
            else f"Only {count} movies in the catalog match: {short_filters}.\n"
        ) if short_filters else ""

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
            if short_line:
                yield _sse_event("token", {"text": short_line})
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

            model_intro = "".join(intro_parts)
            intro = short_line + model_intro
            if not slot_parts and model_intro.strip():
                # No markers at all: every description landed in the intro. The
                # client replaces the streamed text with the opening line and the
                # cards fall back to catalog copy; history gets the same line.
                intro = short_line + _first_line(model_intro)
                yield _sse_event("intro_final", {"text": intro})
            assistant_text = _assistant_turn_text(intro, slot_parts)
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
                        movie_ids=top_ids,
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

    async def _unsupported_response(self, session, message, intent, intent_dict, unchecked, classify_parse_ms):
        """The request names something the catalog cannot check: ask whether to
        search without it instead of showing movies that ignore it.

        No retrieval, no LLM. The stored assistant turn carries
        `pending_relaxation`, so the user's "yes" next turn searches even if
        the classifier lists the same constraint again.
        """
        try:
            await _save_session(session, message, intent_dict, [])
        except Exception:
            logger.exception(f"ChatView {intent.category} error")
            return JsonResponse({"error": "internal error"}, status=500)
        text = UNSUPPORTED_MESSAGE.format(
            constraints=", ".join(f'"{c}"' for c in unchecked),
            unchecked=settings.CATALOG_UNCHECKED_TEXT,
        )

        async def event_stream():
            yield _sse_event("movies", {
                "session_id": str(session.session_id),
                "session_token": session.session_token,
                "movies": [],
                "no_match": False,
                "intent": intent_dict,
            })
            yield _sse_event("token", {"text": text})
            try:
                await _append_history(
                    session, "assistant", text, count_turn=False,
                    pending_relaxation={"dropped": unchecked},
                )
            except Exception:
                logger.exception("failed to store assistant turn")
            _log_turn_latency(intent.category, classify_parse_ms=classify_parse_ms)
            yield _sse_event("done", {})

        return StreamingHttpResponse(
            event_stream(),
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    def _no_match_response(self, session, intent, intent_dict, classify_parse_ms, embed_ms, score_ms):
        """The hard filters or the relevance gate left nothing: say so instead
        of explaining top-5.

        The movies event carries `no_match: true` and an empty list, the fixed
        message (naming the active filters) goes out as a normal token reply,
        and the stored assistant turn has no `movies`/`movie_facts` -- nothing
        was shown, so a later follow-up must not be grounded in movies the user
        never saw.
        """
        text = _no_match_message(intent_dict)

        async def event_stream():
            yield _sse_event("movies", {
                "session_id": str(session.session_id),
                "session_token": session.session_token,
                "movies": [],
                "no_match": True,
                "intent": intent_dict,
            })
            yield _sse_event("token", {"text": text})
            try:
                await _append_history(session, "assistant", text, count_turn=False)
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
            system_prompt = CLARIFY_SYSTEM_PROMPT if category == "clarify" else CONVERSATIONAL_SYSTEM_PROMPT
            async for token in astream_conversational(message, history, system_prompt=system_prompt):
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
