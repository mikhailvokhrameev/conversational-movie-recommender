"""Ollama LLM client for intent parsing and recommendation explanations.

Communicates with a standalone Ollama container via its HTTP API.
Two roles: (1) parse natural language queries into structured intent (JSON mode),
(2) generate Russian-language explanations for recommended movies (RAG pattern).
Falls back to empty intent if Ollama is unavailable (semantic search still works).

The async chat view uses a single combined classify+parse call
(`aclassify_and_parse`, validated against the `MessageIntent` Pydantic schema);
sync management commands still use the separate `parse_intent` call.

Provides both sync and async variants for use in sync management commands
and async Django views respectively.

Hybrid reasoning models (RuadaptQwen3-8B-Hybrid, the configured default) emit
a <think>...</think> span before their answer unless told not to. Every call
here sends Ollama's "think" flag from params.yaml, and both the JSON and the
streaming paths strip any reasoning span that arrives anyway -- so an Ollama
build that ignores the flag costs latency rather than producing broken JSON
or leaking reasoning into the user-visible stream.
"""

import json
import logging
from typing import Generator, Literal, Optional

import httpx
import numpy as np
from asgiref.sync import sync_to_async
from django.conf import settings
from pydantic import BaseModel, Field, ValidationError, field_validator

from core.embedding_service import cosine_similarity, encode_texts

logger = logging.getLogger(__name__)

_genre_embeddings = None

CATALOG_GENRES = [
    "Аниме", "Артхаус", "Биографии", "Блоги", "Боевики", "Вестерны",
    "Военное", "Детективы", "Документальное", "Драмы", "Интервью",
    "Историческое", "Комедии", "Концерты", "Короткий метр", "Криминальное",
    "Курсы", "Мелодрамы", "Музыкальное", "Мультфильмы", "Презентации",
    "Приключения", "Природа", "Путешествия", "Семейное", "Советское",
    "Триллеры", "Ужасы", "Фантастика", "Фильмы для детей", "Фитнес", "Фэнтези",
]

CLASSIFY_AND_PARSE_PROMPT = """You are a message classifier and intent parser for a Russian movie recommendation chatbot. Given a user message, return ONE JSON object with all of the fields below.

Categories (put the best match in "category"):
- "new_search": user wants movie recommendations (e.g. "хочу комедию", "покажи триллеры", "что посмотреть")
- "follow_up": user asks about previously recommended movies (e.g. "расскажи про первый", "кто снял этот фильм?", "о чём он?")
- "refinement": user wants to adjust the last recommendations (e.g. "а повеселее?", "без сериалов", "только российские", "что-нибудь поновее")
- "general_chat": greetings, thanks, questions about the bot (e.g. "привет", "спасибо", "как ты работаешь?")

ALLOWED GENRES (use ONLY these exact strings, copy-paste):
{genres}

JSON fields:
- "category": one of the four categories above
- "semantic_query": the core of what the user is looking for, in Russian, with filler words and pure filter phrases (genre/country/age/year constraints, already captured below) stripped out -- this gets embedded for semantic search, so keep it focused on mood/theme/subject. If category is not new_search/refinement, just repeat the message.
- "genres": list of matching genres from the ALLOWED list above
- "mood": one of: happy, sad, excited, relaxed, romantic, thoughtful, scared, energetic, or ""
- "themes": list of themes mentioned
- "negations": list of genres the user does NOT want (from ALLOWED list)
- "reference_films": list of film titles mentioned
- "country_exclusions": list of countries the user does NOT want (e.g. "США", "Россия", "Франция" -- use the country name as commonly written in Russian, not a genre)
- "max_age_rating": if the user wants something suitable for a specific age or younger (e.g. "для детей" -> 6, "детям можно" -> 12), the maximum age rating as a number, else null
- "min_release_year": if the user wants recent/newer films (e.g. "поновее", "после 2015", "современный") a minimum release year as a number, else null

Example:
User: "хочу что-то смешное, но не ужасы"
{{"category": "new_search", "semantic_query": "весёлый фильм", "genres": ["Комедии"], "mood": "happy", "themes": [], "negations": ["Ужасы"], "reference_films": [], "country_exclusions": [], "max_age_rating": null, "min_release_year": null}}

Example:
User: "расскажи про первый фильм"
{{"category": "follow_up", "semantic_query": "расскажи про первый фильм", "genres": [], "mood": "", "themes": [], "negations": [], "reference_films": [], "country_exclusions": [], "max_age_rating": null, "min_release_year": null}}

Example:
User: "привет"
{{"category": "general_chat", "semantic_query": "привет", "genres": [], "mood": "", "themes": [], "negations": [], "reference_films": [], "country_exclusions": [], "max_age_rating": null, "min_release_year": null}}

Example:
User: "детектив, но не американский, и чтобы поновее"
{{"category": "new_search", "semantic_query": "детектив", "genres": ["Детективы"], "mood": "", "themes": [], "negations": [], "reference_films": [], "country_exclusions": ["США"], "max_age_rating": null, "min_release_year": 2015}}

Now parse this message:
User: "{message}"
"""

CONVERSATIONAL_PROMPT = """You are a Russian-speaking movie recommendation assistant. You ONLY discuss movies, series, directors, actors, genres, and cinema.
{context}

Rules:
- Always respond in Russian
- If the user asks about anything unrelated to movies or cinema, politely redirect: say you are a movie assistant and suggest discussing films instead
- Be concise, friendly, and knowledgeable about cinema

User: "{message}"
"""

INTENT_PROMPT = """You are a movie intent parser. Given a user query, return a JSON object.

ALLOWED GENRES (use ONLY these exact strings, copy-paste):
{genres}

JSON fields:
- "genres": list of matching genres from the ALLOWED list above
- "mood": one of: happy, sad, excited, relaxed, romantic, thoughtful, scared, energetic, or ""
- "themes": list of themes mentioned
- "negations": list of genres the user does NOT want (from ALLOWED list)
- "reference_films": list of film titles mentioned
- "country_exclusions": list of countries the user does NOT want (e.g. "США", "Россия", "Франция" -- use the country name as commonly written in Russian, not a genre)
- "max_age_rating": if the user wants something suitable for a specific age or younger (e.g. "для детей" -> 6, "детям можно" -> 12), the maximum age rating as a number, else null
- "min_release_year": if the user wants recent/newer films (e.g. "поновее", "после 2015", "современный") a minimum release year as a number, else null

Example:
User: "хочу что-то смешное, но не ужасы"
{{"genres": ["Комедии"], "mood": "happy", "themes": [], "negations": ["Ужасы"], "reference_films": [], "country_exclusions": [], "max_age_rating": null, "min_release_year": null}}

Example:
User: "триллер как Молчание ягнят"
{{"genres": ["Триллеры"], "mood": "excited", "themes": [], "negations": [], "reference_films": ["Молчание ягнят"], "country_exclusions": [], "max_age_rating": null, "min_release_year": null}}

Example:
User: "детектив, но не американский, и чтобы поновее"
{{"genres": ["Детективы"], "mood": "", "themes": [], "negations": [], "reference_films": [], "country_exclusions": ["США"], "max_age_rating": null, "min_release_year": 2015}}

Example:
User: "мультфильм для детей"
{{"genres": ["Мультфильмы"], "mood": "", "themes": [], "negations": [], "reference_films": [], "country_exclusions": [], "max_age_rating": 6, "min_release_year": null}}

Now parse this query:
User: "{query}"
"""

EXPLANATION_PROMPT = """You are a movie recommendation assistant speaking Russian. The user asked: "{query}"

Based on their preferences, here are recommended movies. For each movie, write 1-2 sentences in Russian explaining why it matches what the user is looking for. Be specific about the connection between the user's request and each movie's qualities.

Movies:
{movies_context}

Write a brief, natural response in Russian recommending these movies with personalized explanations."""


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _strip_think(text: str) -> str:
    """Remove complete <think>...</think> spans from a finished response.

    An unterminated opening tag means the model was still reasoning when it
    ran out of tokens; everything from it on is dropped.
    """
    while True:
        start = text.find(_THINK_OPEN)
        if start == -1:
            return text
        end = text.find(_THINK_CLOSE, start)
        if end == -1:
            return text[:start]
        text = text[:start] + text[end + len(_THINK_CLOSE):]


def _parse_json_response(content: str) -> dict:
    """Parse an Ollama JSON-mode response, tolerating a reasoning preamble."""
    return json.loads(_strip_think(content).strip())


class _ThinkStreamFilter:
    """Drops <think> spans from a token stream, across chunk boundaries.

    Tags can be split between chunks, so a tail that could still turn into an
    opening tag is held back rather than emitted.
    """

    def __init__(self):
        self._pending = ""
        self._in_think = False

    def feed(self, chunk: str) -> str:
        self._pending += chunk
        out = []

        while self._pending:
            if self._in_think:
                end = self._pending.find(_THINK_CLOSE)
                if end == -1:
                    # Keep only enough to recognise a split closing tag.
                    self._pending = self._pending[-(len(_THINK_CLOSE) - 1):]
                    break
                self._pending = self._pending[end + len(_THINK_CLOSE):]
                self._in_think = False
                continue

            start = self._pending.find(_THINK_OPEN)
            if start != -1:
                out.append(self._pending[:start])
                self._pending = self._pending[start + len(_THINK_OPEN):]
                self._in_think = True
                continue

            # No tag present. Emit everything except a tail that might be the
            # beginning of one.
            hold = 0
            for size in range(min(len(_THINK_OPEN) - 1, len(self._pending)), 0, -1):
                if _THINK_OPEN.startswith(self._pending[-size:]):
                    hold = size
                    break
            if hold:
                out.append(self._pending[:-hold])
                self._pending = self._pending[-hold:]
            else:
                out.append(self._pending)
                self._pending = ""
            break

        return "".join(out)

    def flush(self) -> str:
        """Emit anything held back once the stream ends."""
        if self._in_think:
            return ""
        remainder, self._pending = self._pending, ""
        return remainder


def _build_movies_context(movies: list[dict]) -> str:
    return "\n".join(
        f"- {m.get('serial_name', m.get('title', ''))} ({', '.join(m.get('genres', []))}): {m.get('description', '')[:200]}"
        for m in movies
    )


def check_explanation_titles(
    explanation: str, top_titles: list[str], candidate_titles: list[str]
) -> list[str]:
    """Flag candidate titles the explanation mentions that weren't offered to the user.

    Cheap diagnostic, not a runtime guard: it reuses the retrieval candidate
    pool already held in memory for this turn instead of scanning the full
    catalog, so it only catches the model confusing movies it saw during
    retrieval -- not a title invented outright. Log-only; callers should not
    alter the already-streamed response based on this.
    """
    top_set = set(top_titles)
    lowered = explanation.lower()
    return [
        title for title in candidate_titles
        if title not in top_set and title.lower() in lowered
    ]


def _coerce_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class MessageIntent(BaseModel):
    """Structured output of `aclassify_and_parse`: classification + parsed filters.

    `category` has no default -- a missing or hallucinated value fails
    validation, which is what triggers the repair retry in
    `aclassify_and_parse`. Everything else degrades tolerantly (unknown shape
    -> empty/None), matching the pre-merge `_extract_intent` behavior, since
    losing a filter is far less costly than losing the routing decision.
    """

    category: Literal["new_search", "follow_up", "refinement", "general_chat"]
    semantic_query: str = ""
    genres: list[str] = Field(default_factory=list)
    mood: str = ""
    themes: list[str] = Field(default_factory=list)
    negations: list[str] = Field(default_factory=list)
    reference_films: list[str] = Field(default_factory=list)
    country_exclusions: list[str] = Field(default_factory=list)
    max_age_rating: Optional[float] = None
    min_release_year: Optional[int] = None

    @field_validator("genres", "themes", "negations", "reference_films", mode="before")
    @classmethod
    def _default_to_list(cls, value):
        if not isinstance(value, list):
            return []
        return [v for v in value if isinstance(v, str)]

    @field_validator("country_exclusions", mode="before")
    @classmethod
    def _clean_country_exclusions(cls, value):
        if not isinstance(value, list):
            return []
        return [c for c in value if isinstance(c, str) and c.strip()]

    @field_validator("max_age_rating", mode="before")
    @classmethod
    def _coerce_age_rating(cls, value):
        return _coerce_float(value)

    @field_validator("min_release_year", mode="before")
    @classmethod
    def _coerce_year(cls, value):
        return _coerce_int(value)


def _extract_intent(parsed: dict) -> dict:
    return {
        "genres": _normalize_genres(parsed.get("genres", [])),
        "mood": parsed.get("mood", ""),
        "themes": parsed.get("themes", []),
        "negations": _normalize_genres(parsed.get("negations", [])),
        "reference_films": parsed.get("reference_films", []),
        "country_exclusions": [
            c for c in parsed.get("country_exclusions", []) if isinstance(c, str) and c.strip()
        ],
        "max_age_rating": _coerce_float(parsed.get("max_age_rating")),
        "min_release_year": _coerce_int(parsed.get("min_release_year")),
    }


def _intent_payload(query: str) -> dict:
    return {
        "model": settings.OLLAMA_MODEL,
        "messages": [
            {"role": "user", "content": INTENT_PROMPT.format(
                query=query, genres=", ".join(CATALOG_GENRES)
            )},
        ],
        "format": "json",
        "stream": False,
        "think": settings.OLLAMA_THINKING,
    }


def _classify_and_parse_payload(message: str, repair_note: str | None = None) -> dict:
    content = CLASSIFY_AND_PARSE_PROMPT.format(
        message=message, genres=", ".join(CATALOG_GENRES)
    )
    if repair_note:
        content += (
            f"\n\nYour previous response was invalid: {repair_note}\n"
            "Return ONLY a single valid JSON object matching the schema above."
        )
    return {
        "model": settings.OLLAMA_MODEL,
        "messages": [{"role": "user", "content": content}],
        "format": "json",
        "stream": False,
        "think": settings.OLLAMA_THINKING,
    }


def _explanation_payload(query: str, movies: list[dict], stream: bool = False) -> dict:
    return {
        "model": settings.OLLAMA_MODEL,
        "messages": [
            {
                "role": "user",
                "content": EXPLANATION_PROMPT.format(
                    query=query, movies_context=_build_movies_context(movies)
                ),
            },
        ],
        "stream": stream,
        "think": settings.OLLAMA_THINKING,
    }


# --- Low-level Ollama call (shared by ollama_client and evaluation) ---


def _chat_sync(messages: list[dict], json_mode: bool = False, timeout: float = 60.0) -> str:
    """Send a chat request to Ollama and return the response content string."""
    payload = {
        "model": settings.OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "think": settings.OLLAMA_THINKING,
    }
    if json_mode:
        payload["format"] = "json"
    response = httpx.post(
        f"{settings.OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


# --- Sync API (for management commands) ---


def parse_intent(query: str) -> dict:
    """Extract structured intent from a natural language movie query via Ollama JSON mode."""
    try:
        content = _chat_sync(
            _intent_payload(query)["messages"],
            json_mode=True,
            timeout=settings.OLLAMA_TIMEOUTS["intent"],
        )
        return _extract_intent(_parse_json_response(content))
    except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Intent parsing failed, using fallback: {e}")
        return _fallback_intent(query)


def generate_explanation(query: str, movies: list[dict]) -> str:
    """Generate a Russian-language explanation of why each movie matches the query (RAG)."""
    try:
        return _chat_sync(
            _explanation_payload(query, movies)["messages"],
            timeout=settings.OLLAMA_TIMEOUTS["explanation"],
        )
    except (httpx.HTTPError, KeyError) as e:
        logger.warning(f"Explanation generation failed: {e}")
        return ""


def stream_explanation(query: str, movies: list[dict]) -> Generator[str, None, None]:
    """Streaming variant of generate_explanation. Yields tokens as they arrive."""
    try:
        with httpx.stream(
            "POST",
            f"{settings.OLLAMA_BASE_URL}/api/chat",
            json=_explanation_payload(query, movies, stream=True),
            timeout=settings.OLLAMA_TIMEOUTS["explanation"],
        ) as response:
            think_filter = _ThinkStreamFilter()
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        visible = think_filter.feed(content)
                        if visible:
                            yield visible
            tail = think_filter.flush()
            if tail:
                yield tail
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        logger.warning(f"Explanation streaming failed: {e}")


def is_available() -> bool:
    try:
        response = httpx.get(f"{settings.OLLAMA_BASE_URL}/", timeout=5.0)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


# --- Async API (for Django async views) ---


async def _request_classify_and_parse(message: str, repair_note: str | None = None) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.OLLAMA_BASE_URL}/api/chat",
            json=_classify_and_parse_payload(message, repair_note=repair_note),
            timeout=settings.OLLAMA_TIMEOUTS["classify_parse"],
        )
        response.raise_for_status()
        return _parse_json_response(response.json()["message"]["content"])


def _fallback_message_intent(message: str) -> MessageIntent:
    return MessageIntent(category="new_search", semantic_query=message)


async def aclassify_and_parse(message: str) -> MessageIntent:
    """Classify the message and extract structured intent in a single call.

    Replaces the old two-call aclassify_message + aparse_intent sequence used
    by the async chat view -- fewer round trips, and classify/parse can no
    longer disagree since they're the same response. On a response that
    fails MessageIntent validation (most commonly a hallucinated/missing
    "category"), retries once with the validation error fed back to the
    model; a request-level failure (network/HTTP/JSON) skips the retry and
    falls back immediately, since repairing a message that never arrived
    can't help. A second failure of any kind falls back to
    category="new_search" with empty filters, same as the old classify
    default -- semantic search still works without parsed intent.
    """
    try:
        raw = await _request_classify_and_parse(message)
    except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Classify+parse request failed, using fallback: {e}")
        return _fallback_message_intent(message)

    try:
        intent = MessageIntent.model_validate(raw)
    except ValidationError as e:
        logger.warning(f"Classify+parse validation failed, retrying with repair: {e}")
        try:
            repaired = await _request_classify_and_parse(message, repair_note=str(e))
            intent = MessageIntent.model_validate(repaired)
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, ValidationError) as e2:
            logger.warning(f"Classify+parse repair failed, using fallback: {e2}")
            return _fallback_message_intent(message)

    intent.genres, intent.negations = await sync_to_async(
        lambda: (_normalize_genres(intent.genres), _normalize_genres(intent.negations)),
        thread_sensitive=False,
    )()
    if not intent.semantic_query.strip():
        intent.semantic_query = message
    return intent


async def astream_conversational(message: str, context: str = ""):
    """Stream a conversational response (no movie search). Yields tokens."""
    prompt = CONVERSATIONAL_PROMPT.format(
        message=message,
        context=context if context else "You are chatting casually. No movie context available.",
    )
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{settings.OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": settings.OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True,
                    "think": settings.OLLAMA_THINKING,
                },
                timeout=settings.OLLAMA_TIMEOUTS["explanation"],
            ) as response:
                response.raise_for_status()
                think_filter = _ThinkStreamFilter()
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            visible = think_filter.feed(content)
                            if visible:
                                yield visible
                tail = think_filter.flush()
                if tail:
                    yield tail
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        logger.warning(f"Conversational streaming failed: {e}")


async def astream_explanation(query: str, movies: list[dict]):
    """Async streaming variant of generate_explanation. Yields tokens as they arrive."""
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{settings.OLLAMA_BASE_URL}/api/chat",
                json=_explanation_payload(query, movies, stream=True),
                timeout=settings.OLLAMA_TIMEOUTS["explanation"],
            ) as response:
                response.raise_for_status()
                think_filter = _ThinkStreamFilter()
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            visible = think_filter.feed(content)
                            if visible:
                                yield visible
                tail = think_filter.flush()
                if tail:
                    yield tail
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        logger.warning(f"Async explanation streaming failed: {e}")


async def ais_available() -> bool:
    """Async variant of is_available."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{settings.OLLAMA_BASE_URL}/", timeout=5.0)
            return response.status_code == 200
    except httpx.HTTPError:
        return False


def _get_genre_embeddings() -> np.ndarray:
    """Encode all catalog genres once, cache for the process lifetime."""
    global _genre_embeddings
    if _genre_embeddings is None:
        _genre_embeddings = encode_texts(CATALOG_GENRES)
        logger.info(f"Cached embeddings for {len(CATALOG_GENRES)} catalog genres")
    return _genre_embeddings


def _normalize_genres(raw_genres: list) -> list[str]:
    """Map LLM genre output to exact catalog genre names via embedding similarity.

    'комедия' -> 'Комедии', 'хоррор' -> 'Ужасы', 'мультик' -> 'Мультфильмы'
    """
    if not raw_genres:
        return []

    genre_embs = _get_genre_embeddings()
    normalized = []

    for raw in raw_genres:
        if not isinstance(raw, str) or not raw.strip():
            continue

        if raw in CATALOG_GENRES:
            normalized.append(raw)
            continue

        raw_emb = encode_texts([raw])[0]
        best_score = -1.0
        best_genre = None
        for i, catalog_emb in enumerate(genre_embs):
            sim = cosine_similarity(raw_emb.tolist(), catalog_emb.tolist())
            if sim > best_score:
                best_score = sim
                best_genre = CATALOG_GENRES[i]

        if best_genre and best_score >= settings.GENRE_MATCH_THRESHOLD:
            if best_genre not in normalized:
                normalized.append(best_genre)
            logger.debug(f"Genre normalized: '{raw}' -> '{best_genre}' (sim={best_score:.3f})")
        else:
            logger.debug(f"Genre dropped: '{raw}' (best='{best_genre}' sim={best_score:.3f})")

    return normalized


def _fallback_intent(_query: str) -> dict:
    """Return empty intent when Ollama is unavailable.

    Semantic search via embeddings still works without parsed intent --
    it just loses metadata filtering. This is preferable to broken
    keyword parsing that silently produces wrong results.
    """
    return {
        "genres": [],
        "mood": "",
        "themes": [],
        "negations": [],
        "reference_films": [],
        "country_exclusions": [],
        "max_age_rating": None,
        "min_release_year": None,
    }
