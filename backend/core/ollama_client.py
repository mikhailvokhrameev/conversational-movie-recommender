"""Ollama LLM client for intent parsing and recommendation explanations.

Communicates with a standalone Ollama container via its HTTP API.
Two roles: (1) parse natural language queries into structured intent (JSON-schema-constrained),
(2) generate English-language explanations for recommended movies (RAG pattern).

The explanation is not one blob: the model is asked for an opening line followed
by one `[[n]]`-marked description per movie, and `ExplanationSplitter` routes the
token stream into those slots so the UI can show each description beside the
movie it belongs to.
Falls back to empty intent if Ollama is unavailable (semantic search still works).

The async chat view uses a single combined classify+parse call
(`aclassify_and_parse`, validated against the `MessageIntent` Pydantic schema);
the separate sync `parse_intent` call uses a different prompt, so
`evaluate_scoring` runs `aclassify_and_parse` instead -- it evaluates what ships.

Provides both sync and async variants for use in sync management commands
and async Django views respectively.

Hybrid reasoning models (Qwen3, the configured default) emit
a <think>...</think> span before their answer unless told not to. Every call
here sends Ollama's "think" flag from params.yaml, and both the JSON and the
streaming paths strip any reasoning span that arrives anyway -- so an Ollama
build that ignores the flag costs latency rather than producing broken JSON
or leaking reasoning into the user-visible stream.
"""

import json
import logging
import re
import time
from typing import Generator, Literal, Optional

import httpx
from asgiref.sync import sync_to_async
from django.conf import settings
from pydantic import BaseModel, Field, ValidationError, field_validator

from core.entity_resolution import EntityResolver, catalog_countries, catalog_languages

logger = logging.getLogger(__name__)

CATALOG_GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
    "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery",
    "Romance", "Science Fiction", "TV Movie", "Thriller", "War", "Western",
]

# One resolver per catalog (params.yaml: intent.resolution.<name>); see
# core/entity_resolution.py. Countries and languages are read from the DB on
# first use, so the async path resolves inside sync_to_async.
RESOLVERS = {
    "genres": EntityResolver("genres", lambda: CATALOG_GENRES, settings.INTENT_RESOLUTION["genres"]),
    "countries": EntityResolver("countries", catalog_countries, settings.INTENT_RESOLUTION["countries"]),
    "languages": EntityResolver("languages", catalog_languages, settings.INTENT_RESOLUTION["languages"]),
}
# Intent field -> the resolver that maps it onto catalog values.
RESOLVED_FIELDS = {
    "genres": "genres",
    "negations": "genres",
    "country_inclusions": "countries",
    "country_exclusions": "countries",
    "original_languages": "languages",
}

CLASSIFY_AND_PARSE_SYSTEM_PROMPT = """You are a message classifier and intent parser for a movie recommendation chatbot. Given the conversation so far and the user's latest message, return ONE JSON object with all of the fields below.

Categories (put the best match in "category"):
- "new_search": user wants movie recommendations and gives at least one thing to search on -- a genre, mood, theme, era, country, language, rating, runtime or a film they liked (e.g. "I want a comedy", "show me thrillers", "funny movies", "movies from the 90s")
- "follow_up": user asks about previously recommended movies (e.g. "tell me about the first one", "who directed that?", "what's it about?")
- "refinement": user wants to adjust the last recommendations (e.g. "something funnier?", "no horror", "only French films", "something newer")
- "general_chat": greetings, thanks, questions about the bot (e.g. "hi", "thanks", "how do you work?")
- "clarify": user wants a recommendation but neither the message nor the earlier conversation gives a single thing to search on (e.g. "movies", "recommend something", "what should I watch"). If earlier messages already name a genre, mood or film, it is new_search or refinement instead

The latest message can be incomplete on its own -- a short reply, a pronoun, a bare agreement like "sure" or "yes". Interpret it in light of the earlier messages of this conversation, and fill in field values (a referenced film, a genre, etc.) from that history when the latest message alone doesn't have them.

ALLOWED GENRES (use ONLY these exact strings, copy-paste):
{genres}

JSON fields:
- "category": one of the five categories above
- "semantic_query": the core of what the user is looking for, with filler words and pure filter phrases (genre/country/rating/year/runtime/language constraints, already captured below) stripped out -- this gets embedded for semantic search, so keep it focused on mood/theme/subject. If the message continues earlier turns, carry over the subject from them too (e.g. the theme of a reference film), still without filter phrases. If category is not new_search/refinement, just repeat the message.
- "standalone_query": the user's latest request rewritten as one self-contained English request that someone who never saw this conversation would understand -- carry over the subject, genres, reference films and constraints from earlier messages that still apply, drop filler ("something", "please", "show me"). If the latest message already stands on its own, copy it minus filler. Leave out anything you put in unsupported_constraints. If category is not new_search/refinement, just repeat the message.
- "genres": list of matching genres from the ALLOWED list above
- "mood": one of: happy, sad, excited, relaxed, romantic, thoughtful, scared, energetic, or ""
- "themes": list of themes mentioned
- "negations": list of genres the user does NOT want (from ALLOWED list)
- "reference_films": list of film titles mentioned
- "country_exclusions": list of production countries the user does NOT want (e.g. "United States of America", "France" -- common English country name, not a genre)
- "country_inclusions": list of production countries the user explicitly wants (e.g. "a French film" -> ["France"])
- "min_vote_average": if the user wants highly-rated/critically-acclaimed films, a minimum rating out of 10 (e.g. "highly rated" -> 7.0, "critically acclaimed" -> 7.5), else null
- "min_release_year": if the user wants recent/newer films (e.g. "something newer", "after 2015", "modern") a minimum release year, else null
- "max_release_year": if the user wants older films or a specific era (e.g. "90s movies" -> 1999, "before 2000" -> 1999) a maximum release year, else null
- "min_runtime": if the user wants a long film (e.g. "a long epic") a minimum runtime in minutes, else null
- "max_runtime": if the user wants a short film (e.g. "something short", "under 90 minutes" -> 90) a maximum runtime in minutes, else null
- "original_languages": list of ISO 639-1 language codes if the user names a specific film-industry language (e.g. "a Korean movie" -> ["ko"], "French cinema" -> ["fr"]), else []
- "unsupported_constraints": conditions in the latest message that the catalog cannot check, in the user's words. The catalog knows genre, mood/theme, country, rating, release year, runtime and language. It does NOT know {unchecked} (e.g. "with Keanu Reeves", "directed by Nolan", "silent", "Oscar winners"). Leave them out of every other field. If the assistant just offered to search without a condition and the user agreed, do not list that condition again. Else []

Example:
User: "I want something funny, but not horror"
{{"category": "new_search", "semantic_query": "a fun film", "standalone_query": "a funny film without horror", "genres": ["Comedy"], "mood": "happy", "themes": [], "negations": ["Horror"], "reference_films": [], "country_exclusions": [], "country_inclusions": [], "min_vote_average": null, "min_release_year": null, "max_release_year": null, "min_runtime": null, "max_runtime": null, "original_languages": [], "unsupported_constraints": []}}

Example:
Earlier: the user said "I liked The Godfather". Latest message: "suggest something similar"
{{"category": "refinement", "semantic_query": "mafia family crime saga like The Godfather", "standalone_query": "crime dramas similar to The Godfather", "genres": ["Crime", "Drama"], "mood": "", "themes": [], "negations": [], "reference_films": ["The Godfather"], "country_exclusions": [], "country_inclusions": [], "min_vote_average": null, "min_release_year": null, "max_release_year": null, "min_runtime": null, "max_runtime": null, "original_languages": [], "unsupported_constraints": []}}

Example:
User: "tell me about the first movie"
{{"category": "follow_up", "semantic_query": "tell me about the first movie", "standalone_query": "tell me about the first movie", "genres": [], "mood": "", "themes": [], "negations": [], "reference_films": [], "country_exclusions": [], "country_inclusions": [], "min_vote_average": null, "min_release_year": null, "max_release_year": null, "min_runtime": null, "max_runtime": null, "original_languages": [], "unsupported_constraints": []}}

Example:
User: "hi"
{{"category": "general_chat", "semantic_query": "hi", "standalone_query": "hi", "genres": [], "mood": "", "themes": [], "negations": [], "reference_films": [], "country_exclusions": [], "country_inclusions": [], "min_vote_average": null, "min_release_year": null, "max_release_year": null, "min_runtime": null, "max_runtime": null, "original_languages": [], "unsupported_constraints": []}}

Example:
User: "recommend something"
{{"category": "clarify", "semantic_query": "recommend something", "standalone_query": "recommend something", "genres": [], "mood": "", "themes": [], "negations": [], "reference_films": [], "country_exclusions": [], "country_inclusions": [], "min_vote_average": null, "min_release_year": null, "max_release_year": null, "min_runtime": null, "max_runtime": null, "original_languages": [], "unsupported_constraints": []}}

Example:
User: "a highly rated 90s Korean thriller"
{{"category": "new_search", "semantic_query": "a thriller", "standalone_query": "a highly rated Korean thriller from the 90s", "genres": ["Thriller"], "mood": "", "themes": [], "negations": [], "reference_films": [], "country_exclusions": [], "country_inclusions": [], "min_vote_average": 7.0, "min_release_year": 1990, "max_release_year": 1999, "min_runtime": null, "max_runtime": null, "original_languages": ["ko"], "unsupported_constraints": []}}

Example:
User: "1950s movies with Keanu Reeves"
{{"category": "new_search", "semantic_query": "a 1950s film", "standalone_query": "movies from the 1950s", "genres": [], "mood": "", "themes": [], "negations": [], "reference_films": [], "country_exclusions": [], "country_inclusions": [], "min_vote_average": null, "min_release_year": 1950, "max_release_year": 1959, "min_runtime": null, "max_runtime": null, "original_languages": [], "unsupported_constraints": ["with Keanu Reeves"]}}

Example:
Earlier: the user asked for short animated documentaries and was shown some. Latest message: "They are not silent"
{{"category": "refinement", "semantic_query": "a short animated documentary", "standalone_query": "short animated documentaries", "genres": ["Animation", "Documentary"], "mood": "", "themes": [], "negations": [], "reference_films": [], "country_exclusions": [], "country_inclusions": [], "min_vote_average": null, "min_release_year": null, "max_release_year": null, "min_runtime": null, "max_runtime": 20, "original_languages": [], "unsupported_constraints": ["silent"]}}
"""

CONVERSATIONAL_SYSTEM_PROMPT = """You are a movie recommendation assistant. You ONLY discuss movies, directors, actors, genres, and cinema.

Rules:
- Always respond in English
- If the user asks about anything unrelated to movies or cinema, politely redirect: say you are a movie assistant and suggest discussing films instead
- Be concise, friendly, and knowledgeable about cinema
- Use the earlier messages in this conversation to understand what the user is referring to, especially short replies like agreements or follow-up questions
- The most recent "(Recommended movies: ...)" block in the conversation lists the movies shown to the user, with year, genres, runtime and description from our catalog. Treat those facts as the source of truth when the user asks about those movies (which is shorter, what year, what it is about)
- Our catalog has no director, cast, awards or box-office data. If asked for a detail that is not in the listed facts, say that detail is not in your catalog data; do not state it as fact. You may add general knowledge only if you clearly mark it as such (for example "from general knowledge, not our catalog: ...")
"""

CLARIFY_SYSTEM_PROMPT = """You are a movie recommendation assistant. The user wants a movie but has not said anything to search on yet.

Rules:
- Always respond in English
- Ask ONE short question about the genre, the mood, or a film they liked
- Do not recommend or name any movie titles
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
- "country_exclusions": list of production countries the user does NOT want (e.g. "United States of America", "France")
- "country_inclusions": list of production countries the user explicitly wants (e.g. "a French film" -> ["France"])
- "min_vote_average": if the user wants highly-rated/critically-acclaimed films, a minimum rating out of 10, else null
- "min_release_year": if the user wants recent/newer films a minimum release year, else null
- "max_release_year": if the user wants older films or a specific era a maximum release year, else null
- "min_runtime": if the user wants a long film a minimum runtime in minutes, else null
- "max_runtime": if the user wants a short film a maximum runtime in minutes, else null
- "original_languages": list of ISO 639-1 language codes if the user names a specific film-industry language, else []

Example:
User: "I want something funny, but not horror"
{{"genres": ["Comedy"], "mood": "happy", "themes": [], "negations": ["Horror"], "reference_films": [], "country_exclusions": [], "country_inclusions": [], "min_vote_average": null, "min_release_year": null, "max_release_year": null, "min_runtime": null, "max_runtime": null, "original_languages": []}}

Example:
User: "a thriller like Silence of the Lambs"
{{"genres": ["Thriller"], "mood": "excited", "themes": [], "negations": [], "reference_films": ["Silence of the Lambs"], "country_exclusions": [], "country_inclusions": [], "min_vote_average": null, "min_release_year": null, "max_release_year": null, "min_runtime": null, "max_runtime": null, "original_languages": []}}

Example:
User: "a highly rated 90s Korean thriller"
{{"genres": ["Thriller"], "mood": "", "themes": [], "negations": [], "reference_films": [], "country_exclusions": [], "country_inclusions": [], "min_vote_average": 7.0, "min_release_year": 1990, "max_release_year": 1999, "min_runtime": null, "max_runtime": null, "original_languages": ["ko"]}}

Example:
User: "a short animated family film"
{{"genres": ["Animation", "Family"], "mood": "", "themes": [], "negations": [], "reference_films": [], "country_exclusions": [], "country_inclusions": [], "min_vote_average": null, "min_release_year": null, "max_release_year": null, "min_runtime": null, "max_runtime": 90, "original_languages": []}}

Now parse this query:
User: "{query}"
"""

EXPLANATION_PROMPT = """You are a movie recommendation assistant. The user asked: "{query}"

Movies:
{movies_context}

Answer in English, in two parts.

PART 1 -- an opening line of one or two short sentences confirming the request and
saying what this set has in common. Do NOT name any film title in the opening line.

PART 2 -- for EACH movie above, in the same order, write its marker ([[1]], [[2]], ...)
alone on a line, then two or three sentences about that one film: what it is about,
and why it fits the request. Write about that film only, and never mention another one.

Format: the first line is the opening line. Then, for each movie, a line containing only
its marker, followed by that movie's description. Nothing goes before, between, or after
these lines. For example, for two movies:

Here are two slow-burning thrillers that keep the tension quiet until the very end.
[[1]]
A detective chases a killer through a rain-soaked city; its patient pacing fits a night
when you want suspense without action scenes.
[[2]]
A small-town mystery told over a single winter, matching your wish for a moody,
unhurried story.

Do not add headings, bullet points, numbering, or a closing remark."""

EXPLANATION_SYSTEM_PROMPT = """You are a movie recommendation assistant.

Use the earlier messages in this conversation for context on what the user has been asking for across turns, especially on a refinement of an earlier search.

Answer in English, in two parts.

PART 1 -- an opening line of one or two short sentences confirming the request and
saying what this set has in common. Do NOT name any film title in the opening line.

PART 2 -- for EACH movie listed in the user's latest message, in the same order, write its marker ([[1]], [[2]], ...)
alone on a line, then two or three sentences about that one film: what it is about,
and why it fits the request. Write about that film only, and never mention another one.

Format: the first line is the opening line. Then, for each movie, a line containing only
its marker, followed by that movie's description. Nothing goes before, between, or after
these lines. For example, for two movies:

Here are two slow-burning thrillers that keep the tension quiet until the very end.
[[1]]
A detective chases a killer through a rain-soaked city; its patient pacing fits a night
when you want suspense without action scenes.
[[2]]
A small-town mystery told over a single winter, matching your wish for a moody,
unhurried story.

Do not add headings, bullet points, numbering, or a closing remark."""

EXPLANATION_TURN_TEMPLATE = """The user asked: "{query}"

Movies:
{movies_context}"""


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
    """Parse an Ollama JSON-schema response, tolerating a reasoning preamble."""
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


_MARKER_RE = re.compile(r"\[\[(\d+)\]\]")
# A tail like '[', '[[', '[[1', '[[12' or '[[1]' could still grow into a marker.
_PARTIAL_MARKER_RE = re.compile(r"\[\[?\d*\]?$")
# A prompt-shape placeholder the model echoed back, e.g. '<opening line>': short
# text in angle brackets on one line. A tail that could still close into one
# is held back, like a partial marker.
_PLACEHOLDER_MAX_CHARS = settings.OLLAMA_PLACEHOLDER_MAX_CHARS
_PLACEHOLDER_RE = re.compile(rf"<[^<>\n]{{1,{_PLACEHOLDER_MAX_CHARS}}}>")
_PARTIAL_PLACEHOLDER_RE = re.compile(rf"<[^<>\n]{{0,{_PLACEHOLDER_MAX_CHARS - 1}}}$")


class ExplanationSplitter:
    """Routes an explanation token stream into per-movie slots, across chunk boundaries.

    The model is asked for an opening line followed by one `[[n]]`-marked
    description per movie. `feed` and `flush` return `(slot, text)` pairs:
    `slot` is None for the opening line preceding the first marker, and a
    0-based index into the movie list afterwards -- the markers themselves are
    1-based, matching `_build_movies_context`.

    A marker can be split between chunks, so a tail that could still turn into
    one is held back rather than emitted -- the same discipline
    `_ThinkStreamFilter` applies to `<think>`.

    Placeholders such as `<opening line>` (the model copying a prompt's shape
    literally) are dropped from every slot, with the same hold-back for a tail
    that could still close into one; whitespace right after a dropped
    placeholder is trimmed like the whitespace after a marker.

    Every way the model can deviate degrades silently instead of raising: an
    out-of-range or already-used marker is consumed but leaves the current slot
    alone, and a response with no markers at all stays whole in the intro slot.
    The UI falls back to catalog copy for a movie it received no description
    for, so mis-slotting text is recoverable where dropping it is not.
    """

    def __init__(self, count: int):
        self._count = count
        self._pending = ""
        self._slot = None
        self._seen = set()
        # Whitespace between a marker and its description is layout, not text.
        # The boundary can fall inside that run, so trimming is a state flag
        # rather than a single lstrip.
        self._trim_leading = False

    def _emit(self, out: list, text: str) -> None:
        pieces = _PLACEHOLDER_RE.split(text)
        for i, piece in enumerate(pieces):
            if i:
                self._trim_leading = True
            self._emit_text(out, piece)

    def _emit_text(self, out: list, text: str) -> None:
        if self._trim_leading:
            text = text.lstrip()
            if not text:
                return
            self._trim_leading = False
        if text:
            out.append((self._slot, text))

    def feed(self, chunk: str) -> list[tuple[int | None, str]]:
        self._pending += chunk
        out = []

        while True:
            match = _MARKER_RE.search(self._pending)
            if match is None:
                break
            # Text before the marker still belongs to the previous slot.
            self._emit(out, self._pending[:match.start()])
            index = int(match.group(1)) - 1
            if 0 <= index < self._count and index not in self._seen:
                self._seen.add(index)
                self._slot = index
            self._pending = self._pending[match.end():]
            self._trim_leading = True

        cut = len(self._pending)
        for partial_re in (_PARTIAL_MARKER_RE, _PARTIAL_PLACEHOLDER_RE):
            partial = partial_re.search(self._pending)
            if partial:
                cut = min(cut, partial.start())
        self._emit(out, self._pending[:cut])
        self._pending = self._pending[cut:]
        return out

    def flush(self) -> list[tuple[int | None, str]]:
        """Emit anything held back once the stream ends.

        A tail that never grew into a marker was ordinary text all along.
        """
        out = []
        self._emit(out, self._pending)
        self._pending = ""
        return out


def _build_movies_context(movies: list[dict]) -> str:
    """Prefix each entry with the same [[n]] marker the model must echo back, so a
    streamed description can be routed to the movie it describes."""
    return "\n".join(
        f"[[{i}]] {m.get('serial_name', m.get('title', ''))} ({', '.join(m.get('genres', []))}): {m.get('description', '')[:settings.OLLAMA_EXPLANATION_DESCRIPTION_CHARS]}"
        for i, m in enumerate(movies, 1)
    )


def _format_movie_facts(facts: list[dict]) -> str:
    """One compact line per shown movie: `1. Title (1999) — Drama, Crime — 142 min: description`.
    Missing year/genres/runtime are left out rather than printed as "None"."""
    lines = []
    for i, fact in enumerate(facts, 1):
        head = f"{i}. {fact.get('title', '')}"
        if fact.get("year"):
            head += f" ({fact['year']})"
        parts = [head]
        if fact.get("genres"):
            parts.append(", ".join(fact["genres"]))
        if fact.get("runtime"):
            parts.append(f"{fact['runtime']} min")
        line = " — ".join(parts)
        if fact.get("description"):
            line += f": {fact['description']}"
        lines.append(line)
    return "\n".join(lines)


def _format_history_entry(entry: dict, max_chars: int, with_facts: bool = False) -> dict | None:
    """Turn one `ChatSession.history` entry into a `messages`-array entry.

    Returns None for an entry with an unrecognized/missing `role` -- this is
    what lets legacy (pre-history-feature) sessions replay cleanly, since
    they may contain shapes this feature didn't originally write.

    `with_facts` replaces the titles-only "(Recommended: ...)" line with the
    entry's `movie_facts` block. The caller sets it for the latest shown set
    only (see `_build_chat_messages`); the block is bounded by the storage-time
    description cap and top_n, so `max_chars` applies to the reply text alone.
    """
    role = entry.get("role")
    if role not in ("user", "assistant"):
        return None
    content = str(entry.get("content", ""))[:max_chars]
    movies = entry.get("movies")
    facts = entry.get("movie_facts")
    # An empty reply with movies (explanation failed after the cards were
    # shown) must not start with a blank gap.
    if with_facts and facts:
        content = f"{content}\n\n(Recommended movies:\n{_format_movie_facts(facts)})".lstrip("\n")
    elif movies:
        content = f"{content}\n\n(Recommended: {', '.join(movies)})".lstrip("\n")
    return {"role": role, "content": content}


def _build_chat_messages(
    history: list[dict],
    system_prompt: str,
    current_content: str,
    max_turns: int,
    max_chars: int,
) -> list[dict]:
    """Build a native Ollama multi-turn `messages` list: system prompt, a
    capped history window of real conversation, then the current turn.

    `max_turns` caps history to the last N user+assistant pairs (2*N raw
    entries) before per-entry filtering, so the window size is predictable
    regardless of how many entries turn out to be malformed/skipped. It is
    never "unlimited": 0 yields no history (settings rejects values < 1).

    Only the most recent entry carrying `movie_facts` is rendered with its
    facts: follow-ups refer to the last shown set, and grounding every past
    set would not fit llm.num_ctx. Older sets keep the titles-only line, and
    legacy entries without `movie_facts` render as they always did.
    """
    # history[-0:] is the WHOLE list in Python, so 0 needs its own case.
    windowed = history[-(max_turns * 2):] if max_turns > 0 else []
    latest_facts_index = next(
        (i for i in range(len(windowed) - 1, -1, -1)
         if windowed[i].get("role") == "assistant" and windowed[i].get("movie_facts")),
        None,
    )
    messages = [{"role": "system", "content": system_prompt}]
    for i, entry in enumerate(windowed):
        formatted = _format_history_entry(entry, max_chars, with_facts=(i == latest_facts_index))
        if formatted:
            messages.append(formatted)
    messages.append({"role": "user", "content": current_content})
    return messages


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

    category: Literal["new_search", "follow_up", "refinement", "general_chat", "clarify"]
    semantic_query: str = ""
    standalone_query: str = ""
    genres: list[str] = Field(default_factory=list)
    mood: str = ""
    themes: list[str] = Field(default_factory=list)
    negations: list[str] = Field(default_factory=list)
    reference_films: list[str] = Field(default_factory=list)
    country_exclusions: list[str] = Field(default_factory=list)
    country_inclusions: list[str] = Field(default_factory=list)
    min_vote_average: Optional[float] = None
    min_release_year: Optional[int] = None
    max_release_year: Optional[int] = None
    min_runtime: Optional[int] = None
    max_runtime: Optional[int] = None
    original_languages: list[str] = Field(default_factory=list)
    # Not a filter: conditions the catalog has no data for (cast, director,
    # silent/black-and-white, awards, studio). ChatView asks before searching
    # without them instead of passing them to the pipeline.
    unsupported_constraints: list[str] = Field(default_factory=list)

    @field_validator("genres", "themes", "negations", "reference_films", mode="before")
    @classmethod
    def _default_to_list(cls, value):
        if not isinstance(value, list):
            return []
        return [v for v in value if isinstance(v, str)]

    @field_validator(
        "country_exclusions", "country_inclusions", "original_languages", "unsupported_constraints",
        mode="before",
    )
    @classmethod
    def _clean_string_list(cls, value):
        if not isinstance(value, list):
            return []
        return [c for c in value if isinstance(c, str) and c.strip()]

    @field_validator("min_vote_average", mode="before")
    @classmethod
    def _coerce_vote_average(cls, value):
        return _coerce_float(value)

    @field_validator(
        "min_release_year", "max_release_year", "min_runtime", "max_runtime", mode="before"
    )
    @classmethod
    def _coerce_year_or_runtime(cls, value):
        return _coerce_int(value)


def _resolve_fields(values: dict) -> dict:
    """Map each RESOLVED_FIELDS value list onto exact catalog values. Sync: the
    country and language catalogs come from the DB on first use."""
    return {
        field: RESOLVERS[name].resolve(values.get(field) or [])
        for field, name in RESOLVED_FIELDS.items()
    }


def _extract_intent(parsed: dict) -> dict:
    return {
        "mood": parsed.get("mood", ""),
        "themes": parsed.get("themes", []),
        "reference_films": parsed.get("reference_films", []),
        "min_vote_average": _coerce_float(parsed.get("min_vote_average")),
        "min_release_year": _coerce_int(parsed.get("min_release_year")),
        "max_release_year": _coerce_int(parsed.get("max_release_year")),
        "min_runtime": _coerce_int(parsed.get("min_runtime")),
        "max_runtime": _coerce_int(parsed.get("max_runtime")),
        **_resolve_fields(parsed),
    }


def _ollama_options(kind: Literal["structured", "generation"]) -> dict:
    """Ollama `options` sent with EVERY request. All calls must use the same
    num_ctx: a differing value makes Ollama reload the model.

    `kind` picks the temperature: "structured" for schema-constrained calls whose output
    is parsed (deterministic, so intent parsing and the eval judge are
    reproducible), "generation" for user-facing free text. The seed is fixed
    for both.
    """
    return {
        "num_ctx": settings.OLLAMA_NUM_CTX,
        "seed": settings.OLLAMA_SEED,
        "temperature": settings.OLLAMA_TEMPERATURE[kind],
    }


def _ollama_schema(model: type[BaseModel], exclude: tuple[str, ...] = ()) -> dict:
    """A Pydantic model's JSON schema, shaped for Ollama's `format` field.

    Ollama turns the schema into a decoding grammar, so the model cannot emit
    an off-schema category or a wrong-typed filter. Two adjustments to what
    Pydantic emits: annotation-only keys (`title`, `default`) are dropped, since
    the grammar has no use for them, and every property is made required, so
    the model always returns the full shape the prompt examples show instead of
    omitting fields (the optional ones are nullable, so "no filter" is still
    expressible). `exclude` drops properties the caller does not want decoded.
    Pydantic validation stays the guard on the other side: the schema constrains
    the output, it does not replace validating it.
    """
    schema = model.model_json_schema()
    assert "$defs" not in schema, "nested models would need $ref resolution"
    properties = {
        name: {k: v for k, v in prop.items() if k not in ("title", "default")}
        for name, prop in schema["properties"].items()
        if name not in exclude
    }
    return {"type": "object", "properties": properties, "required": list(properties)}


# Built once: both are constant for the process.
CLASSIFY_PARSE_SCHEMA = _ollama_schema(MessageIntent)
# The MessageIntent fields that are not search filters: only the combined call
# produces them, and every caller drops them before handing the intent to the
# pipeline. What is left is what `_extract_intent` reads.
NON_FILTER_FIELDS = ("category", "semantic_query", "standalone_query", "unsupported_constraints")
INTENT_SCHEMA = _ollama_schema(MessageIntent, exclude=NON_FILTER_FIELDS)


def _intent_payload(query: str) -> dict:
    return {
        "model": settings.OLLAMA_MODEL,
        "messages": [
            {"role": "user", "content": INTENT_PROMPT.format(
                query=query, genres=", ".join(CATALOG_GENRES)
            )},
        ],
        "format": INTENT_SCHEMA,
        "stream": False,
        "think": settings.OLLAMA_THINKING,
        "options": _ollama_options("structured"),
    }


def _classify_and_parse_payload(
    message: str, history: list[dict], repair_note: str | None = None
) -> dict:
    system_prompt = CLASSIFY_AND_PARSE_SYSTEM_PROMPT.format(
        genres=", ".join(CATALOG_GENRES), unchecked=settings.CATALOG_UNCHECKED_TEXT
    )
    current_content = message
    if repair_note:
        current_content += (
            f"\n\nYour previous response was invalid: {repair_note}\n"
            "Return ONLY a single valid JSON object matching the schema above."
        )
    messages = _build_chat_messages(
        history=history,
        system_prompt=system_prompt,
        current_content=current_content,
        max_turns=settings.SESSION_HISTORY_MAX_TURNS,
        max_chars=settings.SESSION_HISTORY_MAX_CHARS_PER_ENTRY,
    )
    return {
        "model": settings.OLLAMA_MODEL,
        "messages": messages,
        "format": CLASSIFY_PARSE_SCHEMA,
        "stream": False,
        "think": settings.OLLAMA_THINKING,
        "options": _ollama_options("structured"),
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
        "options": _ollama_options("generation"),
    }


# --- Call logging: tokens, latency, model identity ---

# Warn when a prompt fills this share of llm.num_ctx or more. History is capped
# by characters, not tokens (see session.history_max_*), so this is the only
# signal that a prompt is close to being truncated by Ollama.
_NUM_CTX_WARN_SHARE = 0.9


def _ns_to_ms(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value / 1_000_000


def _fmt(value, spec: str = "") -> str:
    return "n/a" if value is None else format(value, spec)


def _log_call_stats(call_type: str, stats, ttft_ms: float | None = None) -> None:
    """One INFO line per Ollama call: token counts and Ollama's own timings.

    `stats` is the non-streamed response body, or the final `done: true` chunk
    of a stream; any field may be missing and shows as n/a. `ttft_ms` is the
    client-side time to first token and only exists for streamed calls. Never
    raises: logging must not be able to fail a request.
    """
    if not isinstance(stats, dict):
        stats = {}
    # prompt_eval_count may cover only the newly evaluated tokens when Ollama
    # reuses a cached prompt prefix, so read it from a cold call.
    prompt_tokens = stats.get("prompt_eval_count")
    if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int):
        prompt_tokens = None
    eval_tokens = stats.get("eval_count")
    if isinstance(eval_tokens, bool) or not isinstance(eval_tokens, int):
        eval_tokens = None
    logger.info(
        "Ollama call=%s prompt_tokens=%s eval_tokens=%s prompt_eval_ms=%s eval_ms=%s ttft_ms=%s",
        call_type,
        _fmt(prompt_tokens),
        _fmt(eval_tokens),
        _fmt(_ns_to_ms(stats.get("prompt_eval_duration")), ".0f"),
        _fmt(_ns_to_ms(stats.get("eval_duration")), ".0f"),
        _fmt(ttft_ms, ".0f"),
    )
    if prompt_tokens is not None and prompt_tokens >= _NUM_CTX_WARN_SHARE * settings.OLLAMA_NUM_CTX:
        logger.warning(
            "Ollama call=%s prompt used %d of %d context tokens (llm.num_ctx); "
            "Ollama truncates beyond it. Lower session.history_max_turns or raise llm.num_ctx.",
            call_type, prompt_tokens, settings.OLLAMA_NUM_CTX,
        )


class _StreamStats:
    """Collects what a streamed call reports: the final `done: true` chunk
    carries the token counts, and the first non-empty content chunk gives the
    client-side time to first token (measured from construction, so create it
    just before sending the request)."""

    def __init__(self) -> None:
        self._started = time.perf_counter()
        self.ttft_ms: float | None = None
        self.final: dict = {}

    def observe(self, data: dict) -> None:
        if self.ttft_ms is None and data.get("message", {}).get("content"):
            self.ttft_ms = (time.perf_counter() - self._started) * 1000
        if data.get("done"):
            self.final = data

    def log(self, call_type: str) -> None:
        _log_call_stats(call_type, self.final, self.ttft_ms)


# The tag behind settings.OLLAMA_MODEL (e.g. "qwen3:8b") is mutable and cannot
# be pinned by digest in `ollama pull`, so the digest actually serving requests
# is logged once per process to make a run traceable. One attempt only: a
# failed lookup costs at most one short timeout, never one per request.
_digest_lookup_attempted = False
_DIGEST_LOOKUP_TIMEOUT_SECONDS = 5.0


def _model_digest(tags) -> str | None:
    """The digest of the configured model in an `/api/tags` body, if listed.

    Type-defensive: the body comes from the network, and this runs on the first
    request of a process, so a malformed shape must yield None, not an exception.
    """
    if not isinstance(tags, dict):
        return None
    models = tags.get("models")
    if not isinstance(models, list):
        return None
    wanted = {settings.OLLAMA_MODEL, f"{settings.OLLAMA_MODEL}:latest"}
    for entry in models:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if isinstance(name, str) and name in wanted:
            digest = entry.get("digest")
            return digest if isinstance(digest, str) else None
    return None


def _report_model_digest(tags) -> None:
    digest = _model_digest(tags)
    if digest:
        logger.info("Ollama model %s digest %s", settings.OLLAMA_MODEL, digest)
    else:
        logger.warning("Ollama model %s not listed by /api/tags; digest unknown", settings.OLLAMA_MODEL)


def _log_model_digest_once() -> None:
    global _digest_lookup_attempted
    if _digest_lookup_attempted:
        return
    _digest_lookup_attempted = True
    try:
        response = httpx.get(
            f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=_DIGEST_LOOKUP_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        _report_model_digest(response.json())
    except Exception as e:  # logging must never fail a request
        logger.warning("Could not read the Ollama model digest: %s", e)


async def _alog_model_digest_once(client: httpx.AsyncClient) -> None:
    """Async variant of `_log_model_digest_once`, reusing the caller's client."""
    global _digest_lookup_attempted
    if _digest_lookup_attempted:
        return
    _digest_lookup_attempted = True
    try:
        response = await client.get(
            f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=_DIGEST_LOOKUP_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        _report_model_digest(response.json())
    except Exception as e:  # logging must never fail a request
        logger.warning("Could not read the Ollama model digest: %s", e)


# --- Low-level Ollama call (shared by ollama_client and evaluation) ---


def _stream_chat_payload(messages: list[dict]) -> dict:
    return {
        "model": settings.OLLAMA_MODEL,
        "messages": messages,
        "stream": True,
        "think": settings.OLLAMA_THINKING,
        "options": _ollama_options("generation"),
    }


def _chat_sync(
    messages: list[dict],
    *,
    call_type: str,
    schema: dict | None = None,
    timeout: float = 60.0,
) -> str:
    """Send a chat request to Ollama and return the response content string.

    `schema` is a JSON schema constraining the reply (Ollama `format`); passing
    one also selects the structured sampling kind. `call_type` labels the
    token/latency log line.
    """
    payload = {
        "model": settings.OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "think": settings.OLLAMA_THINKING,
        "options": _ollama_options("structured" if schema else "generation"),
    }
    if schema:
        payload["format"] = schema
    _log_model_digest_once()
    response = httpx.post(
        f"{settings.OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    _log_call_stats(call_type, body)
    return body["message"]["content"]


# --- Sync API (for management commands) ---


def parse_intent(query: str) -> dict:
    """Extract structured intent from a natural language movie query via an Ollama JSON-schema call."""
    try:
        content = _chat_sync(
            _intent_payload(query)["messages"],
            call_type="intent",
            schema=INTENT_SCHEMA,
            timeout=settings.OLLAMA_TIMEOUTS["intent"],
        )
        return _extract_intent(_parse_json_response(content))
    except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Intent parsing failed, using fallback: {e}")
        return _fallback_intent(query)


def generate_explanation(query: str, movies: list[dict]) -> str:
    """Generate an English-language explanation of why each movie matches the query (RAG).

    The result is marker-delimited: an opening line, then a `[[n]]` line per movie.
    """
    try:
        return _chat_sync(
            _explanation_payload(query, movies)["messages"],
            call_type="explanation",
            timeout=settings.OLLAMA_TIMEOUTS["explanation"],
        )
    except (httpx.HTTPError, KeyError) as e:
        logger.warning(f"Explanation generation failed: {e}")
        return ""


def stream_explanation(query: str, movies: list[dict]) -> Generator[str, None, None]:
    """Streaming variant of generate_explanation. Yields tokens as they arrive.

    The token stream is marker-delimited: an opening line, then a `[[n]]` line per movie.
    """
    try:
        stats = _StreamStats()
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
                    stats.observe(data)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        visible = think_filter.feed(content)
                        if visible:
                            yield visible
            stats.log("explanation")
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


async def _request_classify_and_parse(
    message: str, history: list[dict], repair_note: str | None = None
) -> dict:
    async with httpx.AsyncClient() as client:
        await _alog_model_digest_once(client)
        response = await client.post(
            f"{settings.OLLAMA_BASE_URL}/api/chat",
            json=_classify_and_parse_payload(message, history, repair_note=repair_note),
            timeout=settings.OLLAMA_TIMEOUTS["classify_parse"],
        )
        response.raise_for_status()
        body = response.json()
        _log_call_stats("classify_parse", body)
        return _parse_json_response(body["message"]["content"])


def _fallback_message_intent(message: str) -> MessageIntent:
    return MessageIntent(category="new_search", semantic_query=message, standalone_query=message)


async def aclassify_and_parse(message: str, history: list[dict] | None = None) -> MessageIntent:
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
    history = history or []
    try:
        raw = await _request_classify_and_parse(message, history)
    except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Classify+parse request failed, using fallback: {e}")
        return _fallback_message_intent(message)

    try:
        intent = MessageIntent.model_validate(raw)
    except ValidationError as e:
        logger.warning(f"Classify+parse validation failed, retrying with repair: {e}")
        try:
            repaired = await _request_classify_and_parse(message, history, repair_note=str(e))
            intent = MessageIntent.model_validate(repaired)
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, ValidationError) as e2:
            logger.warning(f"Classify+parse repair failed, using fallback: {e2}")
            return _fallback_message_intent(message)

    resolved = await sync_to_async(
        lambda: _resolve_fields(intent.model_dump(include=set(RESOLVED_FIELDS))),
        thread_sensitive=False,
    )()
    for field, values in resolved.items():
        setattr(intent, field, values)
    if not intent.semantic_query.strip():
        intent.semantic_query = message
    if not intent.standalone_query.strip():
        intent.standalone_query = message
    return intent


async def astream_conversational(
    message: str, history: list[dict] | None = None, system_prompt: str = CONVERSATIONAL_SYSTEM_PROMPT,
):
    """Stream a conversational response (no movie search). Yields tokens."""
    history = history or []
    messages = _build_chat_messages(
        history=history,
        system_prompt=system_prompt,
        current_content=message,
        max_turns=settings.SESSION_HISTORY_MAX_TURNS,
        max_chars=settings.SESSION_HISTORY_MAX_CHARS_PER_ENTRY,
    )
    try:
        stats = _StreamStats()
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{settings.OLLAMA_BASE_URL}/api/chat",
                json=_stream_chat_payload(messages),
                timeout=settings.OLLAMA_TIMEOUTS["explanation"],
            ) as response:
                response.raise_for_status()
                think_filter = _ThinkStreamFilter()
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        stats.observe(data)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            visible = think_filter.feed(content)
                            if visible:
                                yield visible
                stats.log("conversational")
                tail = think_filter.flush()
                if tail:
                    yield tail
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        logger.warning(f"Conversational streaming failed: {e}")


async def astream_explanation(query: str, movies: list[dict], history: list[dict] | None = None):
    """Async streaming variant of generate_explanation. Yields tokens as they arrive."""
    history = history or []
    messages = _build_chat_messages(
        history=history,
        system_prompt=EXPLANATION_SYSTEM_PROMPT,
        current_content=EXPLANATION_TURN_TEMPLATE.format(
            query=query, movies_context=_build_movies_context(movies)
        ),
        max_turns=settings.SESSION_HISTORY_MAX_TURNS,
        max_chars=settings.SESSION_HISTORY_MAX_CHARS_PER_ENTRY,
    )
    try:
        stats = _StreamStats()
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{settings.OLLAMA_BASE_URL}/api/chat",
                json=_stream_chat_payload(messages),
                timeout=settings.OLLAMA_TIMEOUTS["explanation"],
            ) as response:
                response.raise_for_status()
                think_filter = _ThinkStreamFilter()
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        stats.observe(data)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            visible = think_filter.feed(content)
                            if visible:
                                yield visible
                stats.log("explanation")
                tail = think_filter.flush()
                if tail:
                    yield tail
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        logger.warning(f"Async explanation streaming failed: {e}")


async def astream_explanation_parts(query: str, movies: list[dict], history: list[dict] | None = None):
    """Stream the explanation already routed into slots. Yields `(slot, text)`.
    `history` is forwarded to `astream_explanation` for conversation context.

    `slot` is None for the opening line and a 0-based index into `movies` for
    each per-movie description -- see `ExplanationSplitter`. `astream_explanation`
    is deliberately left as a raw token generator so the splitting stays
    testable on its own, without an HTTP stream in the way.
    """
    splitter = ExplanationSplitter(len(movies))
    async for token in astream_explanation(query, movies, history):
        for part in splitter.feed(token):
            yield part
    for part in splitter.flush():
        yield part


async def ais_available() -> bool:
    """Async variant of is_available."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{settings.OLLAMA_BASE_URL}/", timeout=5.0)
            return response.status_code == 200
    except httpx.HTTPError:
        return False


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
        "country_inclusions": [],
        "min_vote_average": None,
        "min_release_year": None,
        "max_release_year": None,
        "min_runtime": None,
        "max_runtime": None,
        "original_languages": [],
    }
