"""Common LLM client for intent parsing and recommendation explanations.

Provider protocols live in llm_transport; dependency assembly selects the adapter.
Two roles: (1) parse natural language queries into structured intent (JSON-schema-constrained),
(2) generate English-language explanations for recommended movies (RAG pattern).

The explanation is not one blob: the model is asked for an opening line followed
by one `[[n]]`-marked description per movie, and `ExplanationSplitter` routes the
token stream into those slots so the UI can show each description beside the
movie it belongs to.
Falls back to empty intent if the configured provider is unavailable (semantic search still works).

The async chat view uses a single combined classify+parse call
(`aclassify_and_parse`, validated against the `MessageIntent` Pydantic schema);
the separate sync `parse_intent` call uses a different prompt, so
`evaluate_scoring` runs `aclassify_and_parse` instead -- it evaluates what ships.

Provides both sync and async variants for use in sync management commands
and async Django views respectively.

Hybrid reasoning models (Qwen3, the configured default) emit
a <think>...</think> span before their answer unless told not to. Every call
uses the adapter's reasoning flag from params.yaml, and both JSON parsing and
stream filtering strip any reasoning span that arrives anyway.
"""

import json
import logging
import re
from typing import Generator, Literal, Optional

import httpx
from asgiref.sync import sync_to_async
from django.conf import settings
from pydantic import BaseModel, Field, ValidationError, field_validator

from core.catalog_bounds import catalog_year_range
from core.inference_backends import get_inference_backends
from core.openrouter_client import InferenceError
from core.inference_diagnostics import inference_operation, record_failure
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
- "refinement": user asks for different movies or gives a new condition to adjust the last recommendations (e.g. "something funnier?", "no horror", "only French films", "something newer"). Not when the user only reports that the previous ones did not fit -- that is general_chat
- "general_chat": greetings, thanks, questions about the bot, or complaints about its previous answer (e.g. "hi", "thanks", "how do you work?", "why only one movie?", "I asked for X, why...")
- "clarify": user wants a recommendation but neither the message nor the earlier conversation gives a single thing to search on (e.g. "movies", "recommend something", "what should I watch"). If earlier messages already name a genre, mood or film, it is new_search or refinement instead

The latest message can be incomplete on its own -- a short reply, a pronoun, a bare agreement like "sure" or "yes". Interpret it in light of the earlier messages of this conversation, and fill in field values (a referenced film, a genre, etc.) from that history when the latest message alone doesn't have them.

If the latest message ends with a note like "[The user is viewing movie N of the latest list: Title (year)]", that is the film the user has open. "this", "it", "this one" and "this movie" mean that film: a question about it is a "follow_up"; "more like this" is a "refinement" with that title in "reference_films". A message that names another film or points to another position ("the second one") is not about the note. The note is context, not part of the request -- never copy it into the queries.

ALLOWED GENRES (use ONLY these exact strings, copy-paste):
{genres}

JSON fields:
- "category": one of the five categories above
- "semantic_query": the core of what the user is looking for, with filler words and pure filter phrases (genre/country/rating/year/runtime/language constraints, already captured below) stripped out -- this gets embedded for semantic search, so keep it focused on mood/theme/subject. If the message continues earlier turns, carry over the subject from them too (e.g. the theme of a reference film), still without filter phrases. If category is not new_search/refinement, just repeat the message.
- "standalone_query": the user's latest request rewritten as one self-contained English request that someone who never saw this conversation would understand -- carry over the subject, genres, reference films and constraints from earlier messages that still apply, drop filler ("something", "please", "show me"). If the latest message already stands on its own, copy it minus filler. Leave out anything you put in unsupported_constraints. If category is not new_search/refinement, just repeat the message.
- "genres": list of matching genres from the ALLOWED list above
- "mood": one of: happy, sad, excited, relaxed, romantic, thoughtful, scared, energetic, or ""
- "themes": list of themes mentioned
- "negations": genres the user explicitly rejects by name (from ALLOWED list); "not about cars" is a theme, not a genre negation
- "reference_films": list of film titles mentioned
- "country_exclusions": list of production countries the user does NOT want (e.g. "United States of America", "France" -- common English country name, not a genre)
- "country_inclusions": list of production countries the user explicitly wants (e.g. "a French film" -> ["France"])
- "min_vote_average": if the user wants highly-rated/critically-acclaimed films, a minimum rating out of 10 (e.g. "highly rated" -> 7.0, "critically acclaimed" -> 7.5), else null
- "min_release_year": if the user wants recent/newer films (e.g. "something newer", "after 2015", "modern") a minimum release year, else null{year_rule}
- "max_release_year": if the user wants older films or a specific era (e.g. "90s movies" -> 1999, "before 2000" -> 1999) a maximum release year, else null. "classic" or "old-school" without an era sets no year
- "min_runtime": if the user wants a long film (e.g. "a long epic") a minimum runtime in minutes, else null
- "max_runtime": if the user wants a short film (e.g. "something short", "under 90 minutes" -> 90) a maximum runtime in minutes, else null
- "original_languages": list of ISO 639-1 language codes if the user names a specific film-industry language (e.g. "a Korean movie" -> ["ko"], "French cinema" -> ["fr"]), else []
- "unsupported_constraints": conditions in the latest message that the catalog cannot check, in the user's words. The catalog knows genre, mood/theme, country, rating, release year, runtime and language. It does NOT know {unchecked} (e.g. "with Keanu Reeves", "directed by Nolan", "in the style of Nolan", "silent", "Oscar winners"). A named director, a named actor or an award/prize is always unsupported, and a director named as a style or influence ("in the style of ...") is still a named director: list it here and keep it out of semantic_query and standalone_query, even when the rest of the message is searchable. Leave them out of every other field. If the assistant just offered to search without a condition and the user agreed, do not list that condition again. Else []
- "result_count": the number of movies the user asks to be shown ("only one", "just 1 more" -> 1, "top 10" -> 10), else null
- "wants_obscure": true only if the user explicitly asks for obscure, little-known, weird, cult-trash or low-rated films, else false

Example:
User: "I want something funny, but not horror"
{{"category": "new_search", "semantic_query": "a fun film", "standalone_query": "a funny film without horror", "genres": ["Comedy"], "mood": "happy", "themes": [], "negations": ["Horror"], "reference_films": [], "country_exclusions": [], "country_inclusions": [], "min_vote_average": null, "min_release_year": null, "max_release_year": null, "min_runtime": null, "max_runtime": null, "original_languages": [], "unsupported_constraints": [], "result_count": null, "wants_obscure": false}}

Example:
Earlier: the user said "I liked The Godfather". Latest message: "suggest something similar"
{{"category": "refinement", "semantic_query": "mafia family crime saga like The Godfather", "standalone_query": "crime dramas similar to The Godfather", "genres": ["Crime", "Drama"], "mood": "", "themes": [], "negations": [], "reference_films": ["The Godfather"], "country_exclusions": [], "country_inclusions": [], "min_vote_average": null, "min_release_year": null, "max_release_year": null, "min_runtime": null, "max_runtime": null, "original_languages": [], "unsupported_constraints": [], "result_count": null, "wants_obscure": false}}

Example:
User: "tell me about the first movie"
{{"category": "follow_up", "semantic_query": "tell me about the first movie", "standalone_query": "tell me about the first movie", "genres": [], "mood": "", "themes": [], "negations": [], "reference_films": [], "country_exclusions": [], "country_inclusions": [], "min_vote_average": null, "min_release_year": null, "max_release_year": null, "min_runtime": null, "max_runtime": null, "original_languages": [], "unsupported_constraints": [], "result_count": null, "wants_obscure": false}}

Example:
User: "hi"
{{"category": "general_chat", "semantic_query": "hi", "standalone_query": "hi", "genres": [], "mood": "", "themes": [], "negations": [], "reference_films": [], "country_exclusions": [], "country_inclusions": [], "min_vote_average": null, "min_release_year": null, "max_release_year": null, "min_runtime": null, "max_runtime": null, "original_languages": [], "unsupported_constraints": [], "result_count": null, "wants_obscure": false}}

Example:
User: "recommend something"
{{"category": "clarify", "semantic_query": "recommend something", "standalone_query": "recommend something", "genres": [], "mood": "", "themes": [], "negations": [], "reference_films": [], "country_exclusions": [], "country_inclusions": [], "min_vote_average": null, "min_release_year": null, "max_release_year": null, "min_runtime": null, "max_runtime": null, "original_languages": [], "unsupported_constraints": [], "result_count": null, "wants_obscure": false}}

Example:
User: "a highly rated 90s Korean thriller"
{{"category": "new_search", "semantic_query": "a thriller", "standalone_query": "a highly rated Korean thriller from the 90s", "genres": ["Thriller"], "mood": "", "themes": [], "negations": [], "reference_films": [], "country_exclusions": [], "country_inclusions": [], "min_vote_average": 7.0, "min_release_year": 1990, "max_release_year": 1999, "min_runtime": null, "max_runtime": null, "original_languages": ["ko"], "unsupported_constraints": [], "result_count": null, "wants_obscure": false}}

Example:
User: "1950s movies with Keanu Reeves"
{{"category": "new_search", "semantic_query": "a 1950s film", "standalone_query": "movies from the 1950s", "genres": [], "mood": "", "themes": [], "negations": [], "reference_films": [], "country_exclusions": [], "country_inclusions": [], "min_vote_average": null, "min_release_year": 1950, "max_release_year": 1959, "min_runtime": null, "max_runtime": null, "original_languages": [], "unsupported_constraints": ["with Keanu Reeves"], "result_count": null, "wants_obscure": false}}

Example:
User: "a moody sci-fi directed by Ridley Scott"
{{"category": "new_search", "semantic_query": "a moody science fiction film", "standalone_query": "a moody science fiction film", "genres": ["Science Fiction"], "mood": "thoughtful", "themes": [], "negations": [], "reference_films": [], "country_exclusions": [], "country_inclusions": [], "min_vote_average": null, "min_release_year": null, "max_release_year": null, "min_runtime": null, "max_runtime": null, "original_languages": [], "unsupported_constraints": ["directed by Ridley Scott"], "result_count": null, "wants_obscure": false}}

Example:
User: "a mind-bending sci-fi thriller in the style of Nolan"
{{"category": "new_search", "semantic_query": "a mind-bending science fiction thriller", "standalone_query": "a mind-bending science fiction thriller", "genres": ["Science Fiction", "Thriller"], "mood": "", "themes": [], "negations": [], "reference_films": [], "country_exclusions": [], "country_inclusions": [], "min_vote_average": null, "min_release_year": null, "max_release_year": null, "min_runtime": null, "max_runtime": null, "original_languages": [], "unsupported_constraints": ["in the style of Nolan"], "result_count": null, "wants_obscure": false}}

Example:
User: "a romantic comedy starring Julia Roberts"
{{"category": "new_search", "semantic_query": "a romantic comedy", "standalone_query": "a romantic comedy", "genres": ["Romance", "Comedy"], "mood": "romantic", "themes": [], "negations": [], "reference_films": [], "country_exclusions": [], "country_inclusions": [], "min_vote_average": null, "min_release_year": null, "max_release_year": null, "min_runtime": null, "max_runtime": null, "original_languages": [], "unsupported_constraints": ["starring Julia Roberts"], "result_count": null, "wants_obscure": false}}

Example:
User: "a Palme d'Or winning war drama"
{{"category": "new_search", "semantic_query": "a war drama", "standalone_query": "a war drama", "genres": ["War", "Drama"], "mood": "", "themes": [], "negations": [], "reference_films": [], "country_exclusions": [], "country_inclusions": [], "min_vote_average": null, "min_release_year": null, "max_release_year": null, "min_runtime": null, "max_runtime": null, "original_languages": [], "unsupported_constraints": ["Palme d'Or winning"], "result_count": null, "wants_obscure": false}}

Example:
User: "animated silent documentary under 20 minutes"
{{"category": "new_search", "semantic_query": "an animated documentary", "standalone_query": "animated documentary under 20 minutes", "genres": ["Animation", "Documentary"], "mood": "", "themes": [], "negations": [], "reference_films": [], "country_exclusions": [], "country_inclusions": [], "min_vote_average": null, "min_release_year": null, "max_release_year": null, "min_runtime": null, "max_runtime": 20, "original_languages": [], "unsupported_constraints": ["silent"], "result_count": null, "wants_obscure": false}}

Example:
Earlier: the user asked for a good comedy and was shown some. Latest message: "only 1 more movie"
{{"category": "refinement", "semantic_query": "a good comedy", "standalone_query": "a good comedy", "genres": ["Comedy"], "mood": "", "themes": [], "negations": [], "reference_films": [], "country_exclusions": [], "country_inclusions": [], "min_vote_average": null, "min_release_year": null, "max_release_year": null, "min_runtime": null, "max_runtime": null, "original_languages": [], "unsupported_constraints": [], "result_count": 1, "wants_obscure": false}}

Example:
Earlier: the user asked for short animated documentaries and was shown some. Latest message: "They are not silent"
{{"category": "general_chat", "semantic_query": "They are not silent", "standalone_query": "They are not silent", "genres": [], "mood": "", "themes": [], "negations": [], "reference_films": [], "country_exclusions": [], "country_inclusions": [], "min_vote_average": null, "min_release_year": null, "max_release_year": null, "min_runtime": null, "max_runtime": null, "original_languages": [], "unsupported_constraints": [], "result_count": null, "wants_obscure": false}}

Example:
Earlier: the user asked for a slow western and was shown some. Latest message: "bro I told you I wanted old ones"
{{"category": "general_chat", "semantic_query": "bro I told you I wanted old ones", "standalone_query": "bro I told you I wanted old ones", "genres": [], "mood": "", "themes": [], "negations": [], "reference_films": [], "country_exclusions": [], "country_inclusions": [], "min_vote_average": null, "min_release_year": null, "max_release_year": null, "min_runtime": null, "max_runtime": null, "original_languages": [], "unsupported_constraints": [], "result_count": null, "wants_obscure": false}}
"""

CONVERSATIONAL_SYSTEM_PROMPT = """You are a movie recommendation assistant. You ONLY discuss movies, directors, actors, genres, and cinema.

Rules:
- Always respond in English
- If the user asks about anything unrelated to movies or cinema, politely redirect: say you are a movie assistant and suggest discussing films instead
- Be concise, friendly, and knowledgeable about cinema
- Use the earlier messages in this conversation to understand what the user is referring to, especially short replies like agreements or follow-up questions
- The most recent "(Recommended movies: ...)" block in the conversation lists the movies shown to the user, with year, genres, runtime and description from our catalog. Treat those facts as the source of truth when the user asks about those movies (which is shorter, what year, what it is about)
- When answering about a movie from the "(Recommended movies: ...)" list or a "Selected movie" block, answer only from the catalog facts: title, year, genres, runtime and description from the list, plus rating, countries and original language from the block. Our catalog has no director, cast, actors, awards or box-office data. If asked for anything that is not in those facts (who stars in it, who directed it, which award it won), say that detail is not in your catalog data; never guess it and do not add it from general knowledge. For a film that is not in the list or the block, no catalog facts are supplied: you may use general knowledge only if you clearly mark it as such (for example "from general knowledge, not our catalog: ...")
- If the user complains about the movies just shown (for example "they are not silent", "I told you I wanted old ones"), acknowledge the complaint. If it is about something our catalog cannot check ({unchecked}), say honestly that the catalog cannot verify it and the shown movies were not checked against it. Offer one concrete next step (for example a new search on the parts the catalog can check). Do not give a new list of movies and do not name any film that is not in the listed facts
- If the latest user message is followed by a "Selected movie (full catalog facts)" block, the user has that movie open (it is movie N of the latest list). The block's "Position" line says which movie of the list it is. Use the block when the question refers to it with "this", "it", "this one" or "this movie", or points to the same position ("the first one" when the block says movie 1). A question that points to another film by title or by a different position ("the second one" when the block says movie 1), or that compares or ranks the listed movies ("which one is shorter?"), is answered from the "(Recommended movies: ...)" facts, not from the block
""".format(unchecked=settings.CATALOG_UNCHECKED_TEXT)

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
- "wants_obscure": true only if the user explicitly asks for obscure, little-known, weird, cult-trash or low-rated films, else false

Example:
User: "I want something funny, but not horror"
{{"genres": ["Comedy"], "mood": "happy", "themes": [], "negations": ["Horror"], "reference_films": [], "country_exclusions": [], "country_inclusions": [], "min_vote_average": null, "min_release_year": null, "max_release_year": null, "min_runtime": null, "max_runtime": null, "original_languages": [], "wants_obscure": false}}

Example:
User: "a thriller like Silence of the Lambs"
{{"genres": ["Thriller"], "mood": "excited", "themes": [], "negations": [], "reference_films": ["Silence of the Lambs"], "country_exclusions": [], "country_inclusions": [], "min_vote_average": null, "min_release_year": null, "max_release_year": null, "min_runtime": null, "max_runtime": null, "original_languages": [], "wants_obscure": false}}

Example:
User: "a highly rated 90s Korean thriller"
{{"genres": ["Thriller"], "mood": "", "themes": [], "negations": [], "reference_films": [], "country_exclusions": [], "country_inclusions": [], "min_vote_average": 7.0, "min_release_year": 1990, "max_release_year": 1999, "min_runtime": null, "max_runtime": null, "original_languages": ["ko"], "wants_obscure": false}}

Example:
User: "a short animated family film"
{{"genres": ["Animation", "Family"], "mood": "", "themes": [], "negations": [], "reference_films": [], "country_exclusions": [], "country_inclusions": [], "min_vote_average": null, "min_release_year": null, "max_release_year": null, "min_runtime": null, "max_runtime": 90, "original_languages": [], "wants_obscure": false}}

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
Describe the movies from the listed facts only: never state that the movies meet a
condition (silent, a rating, a year, a country) the listed facts do not show.
If the user's message has a "[Not checked: ...]" field, those conditions were not in the
catalog data and the movies were never checked against them. Say so in your own words in
the opening line (for example "I could not check X, so these are chosen on the rest of
your request"). Never copy the field, its brackets or its label into your answer.

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
{unverified_line}
Movies:
{movies_context}"""

# A labelled data field, not a sentence: the system prompt says how to use it.
# An instruction placed in the user turn got read out as answer text.
EXPLANATION_UNVERIFIED_LINE = "[Not checked: {conditions}]\n"


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
_PLACEHOLDER_MAX_CHARS = settings.LLM_PLACEHOLDER_MAX_CHARS
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
        f"[[{i}]] {m.get('serial_name', m.get('title', ''))} ({', '.join(m.get('genres', []))}): {m.get('description', '')[:settings.LLM_EXPLANATION_DESCRIPTION_CHARS]}"
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


def _coerce_bool(value) -> bool:
    """Tolerant bool: "true"/"yes"/"1" -> True; None and anything unrecognised -> False."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return False


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
    # Not a filter: how many movies the user asks to see; None = no preference
    # (ChatView then uses diversification.top_n and clamps to diversification.max_n).
    result_count: Optional[int] = None
    # Filter-like (reaches the pipeline intent dict): True only when the user
    # explicitly asks for obscure/weird/low-rated films. It switches the
    # popularity-quality signal and the well-known channel off and nothing else.
    wants_obscure: bool = False

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

    @field_validator("wants_obscure", mode="before")
    @classmethod
    def _coerce_wants_obscure(cls, value):
        return _coerce_bool(value)

    @field_validator("min_vote_average", mode="before")
    @classmethod
    def _coerce_vote_average(cls, value):
        return _coerce_float(value)

    @field_validator(
        "min_release_year", "max_release_year", "min_runtime", "max_runtime", "result_count",
        mode="before",
    )
    @classmethod
    def _coerce_year_or_runtime(cls, value):
        return _coerce_int(value)


def _resolve_fields(values: dict) -> dict:
    """Map each RESOLVED_FIELDS value list onto exact catalog values. Sync: the
    country and language catalogs come from the DB on first use."""
    return {
        field: RESOLVERS[name].resolve(values.get(field) or [], field)
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
        "wants_obscure": _coerce_bool(parsed.get("wants_obscure")),
        **_resolve_fields(parsed),
    }


def _llm_schema(model: type[BaseModel], exclude: tuple[str, ...] = ()) -> dict:
    """A Pydantic model's common JSON schema, shared by both adapters.

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
CLASSIFY_PARSE_SCHEMA = _llm_schema(MessageIntent)
# The MessageIntent fields that are not search filters: only the combined call
# produces them, and every caller drops them before handing the intent to the
# pipeline. What is left is what `_extract_intent` reads.
NON_FILTER_FIELDS = (
    "category", "semantic_query", "standalone_query", "unsupported_constraints", "result_count",
)
INTENT_SCHEMA = _llm_schema(MessageIntent, exclude=NON_FILTER_FIELDS)


def _intent_payload(query: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": INTENT_PROMPT.format(
                query=query, genres=", ".join(CATALOG_GENRES)
            )},
        ],
        "schema": INTENT_SCHEMA,
        "kind": "structured",
    }


# Appended to the min_release_year bullet when the catalog's bounds are known.
CATALOG_YEAR_RULE = (
    '. The catalog has films released up to {catalog_max_year}: for "newest", "latest" or "recent" '
    "set it to {recent_min_year} (the catalog's last {recent_years} years), never the calendar year"
)


def _catalog_year_rule(year_bounds: tuple | None) -> str:
    if year_bounds is None:
        return ""
    catalog_max_year = year_bounds[1]
    return CATALOG_YEAR_RULE.format(
        catalog_max_year=catalog_max_year,
        recent_years=settings.INTENT_RECENT_YEARS,
        recent_min_year=catalog_max_year - settings.INTENT_RECENT_YEARS + 1,
    )


def _classify_and_parse_payload(
    message: str, history: list[dict], repair_note: str | None = None, year_bounds: tuple | None = None,
    selection_note: str | None = None,
) -> dict:
    system_prompt = CLASSIFY_AND_PARSE_SYSTEM_PROMPT.format(
        genres=", ".join(CATALOG_GENRES), unchecked=settings.CATALOG_UNCHECKED_TEXT,
        year_rule=_catalog_year_rule(year_bounds),
    )
    current_content = message
    if selection_note:
        current_content += f"\n\n{selection_note}"
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
        "messages": messages,
        "schema": CLASSIFY_PARSE_SCHEMA,
        "kind": "structured",
    }


def _explanation_payload(query: str, movies: list[dict]) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": EXPLANATION_PROMPT.format(
                    query=query, movies_context=_build_movies_context(movies)
                ),
            },
        ],
        "kind": "generation",
        "schema": None,
    }


# --- Low-level completion (shared by llm_client and evaluation) ---


def _stream_chat_payload(messages: list[dict]) -> dict:
    return {
        "messages": messages,
        "kind": "generation",
        "schema": None,
    }


def _chat_sync(
    messages: list[dict], *, call_type: str, schema: dict | None = None,
    timeout: float = 60.0,
) -> str:
    """Return a completion from the configured adapter using common messages."""
    with inference_operation("llm", call_type):
        return get_inference_backends().llm.complete(
            messages=messages, schema=schema,
            kind="structured" if schema is not None else "generation",
            timeout=timeout, call_type=call_type,
        )


# --- Sync API (for management commands) ---


def parse_intent(query: str) -> dict:
    """Extract structured intent from a natural language movie query via a schema-constrained call."""
    try:
        content = _chat_sync(
            _intent_payload(query)["messages"],
            call_type="intent",
            schema=INTENT_SCHEMA,
            timeout=settings.LLM_TIMEOUTS["intent"],
        )
        return _extract_intent(_parse_json_response(content))
    except (httpx.HTTPError, InferenceError, json.JSONDecodeError, KeyError) as e:
        record_failure("llm", "intent", e)
        return _fallback_intent(query)


def generate_explanation(query: str, movies: list[dict]) -> str:
    """Generate an English-language explanation of why each movie matches the query (RAG).

    The result is marker-delimited: an opening line, then a `[[n]]` line per movie.
    """
    try:
        return _chat_sync(
            _explanation_payload(query, movies)["messages"],
            call_type="explanation",
            timeout=settings.LLM_TIMEOUTS["explanation"],
        )
    except (httpx.HTTPError, InferenceError, KeyError) as e:
        record_failure("llm", "explanation", e)
        return ""


def stream_explanation(query: str, movies: list[dict]) -> Generator[str, None, None]:
    """Streaming variant of generate_explanation. Yields tokens as they arrive.

    The token stream is marker-delimited: an opening line, then a `[[n]]` line per movie.
    """
    try:
        with inference_operation("llm", "explanation"):
            think_filter = _ThinkStreamFilter()
            for content in get_inference_backends().llm.stream(
                **_explanation_payload(query, movies),
                timeout=settings.LLM_TIMEOUTS["explanation"], call_type="explanation",
            ):
                visible = think_filter.feed(content)
                if visible:
                    yield visible
            tail = think_filter.flush()
            if tail:
                yield tail
    except (httpx.HTTPError, InferenceError, json.JSONDecodeError) as e:
        record_failure("llm", "explanation", e)


def is_available() -> bool:
    return get_inference_backends().llm.is_available()


# --- Async API (for Django async views) ---


async def _request_classify_and_parse(
    message: str, history: list[dict], repair_note: str | None = None, selection_note: str | None = None
) -> dict:
    year_bounds = await sync_to_async(catalog_year_range, thread_sensitive=False)()
    with inference_operation("llm", "classify_parse"):
        content = await get_inference_backends().llm.acomplete(
            **_classify_and_parse_payload(
                message, history, repair_note=repair_note, year_bounds=year_bounds,
                selection_note=selection_note,
            ), timeout=settings.LLM_TIMEOUTS["classify_parse"], call_type="classify_parse",
        )
        return _parse_json_response(content)


def _fallback_message_intent(message: str) -> MessageIntent:
    return MessageIntent(category="new_search", semantic_query=message, standalone_query=message)


async def aclassify_and_parse(
    message: str, history: list[dict] | None = None, selection_note: str | None = None,
) -> MessageIntent:
    """Classify the message and extract structured intent in a single call.

    Replaces the old two-call aclassify_message + aparse_intent sequence used
    by the async chat view -- fewer round trips, and classify/parse can no
    longer disagree since they're the same response. On a response that
    fails MessageIntent validation (most commonly a hallucinated/missing
    "category"), retries once with the validation error fed back to the
    model; a request-level failure (network/HTTP/JSON) skips the retry and
    falls back immediately, since repairing a message that never arrived
    can't help. `selection_note` is appended to the user message (the card the
    user is viewing, so "this"/"it" resolve). A second failure of any kind falls back to
    category="new_search" with empty filters, same as the old classify
    default -- semantic search still works without parsed intent.
    """
    history = history or []
    try:
        raw = await _request_classify_and_parse(message, history, selection_note=selection_note)
    except (httpx.HTTPError, InferenceError, json.JSONDecodeError, KeyError) as e:
        record_failure("llm", "classify_parse", e)
        logger.warning("Classify+parse request failed, using fallback")
        return _fallback_message_intent(message)

    try:
        intent = MessageIntent.model_validate(raw)
    except ValidationError as e:
        logger.warning("Classify+parse validation failed, retrying with repair")
        try:
            repaired = await _request_classify_and_parse(
                message, history, repair_note=str(e), selection_note=selection_note,
            )
            intent = MessageIntent.model_validate(repaired)
        except (httpx.HTTPError, InferenceError, json.JSONDecodeError, KeyError, ValidationError) as e2:
            record_failure("llm", "classify_parse", e2)
            logger.warning("Classify+parse repair failed, using fallback")
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


def _selected_movie_block(movie: dict, position: int | None = None) -> str:
    """The full catalog record of the card the user is viewing, for the current turn."""
    release_date = movie.get("release_date")
    lines = [
        *([f"Position: movie {position} of the latest list"] if position else []),
        f"Title: {movie['serial_name']}",
        f"Year: {str(release_date)[:4] if release_date else 'unknown'}",
        f"Genres: {', '.join(movie.get('genres') or []) or 'unknown'}",
        f"Runtime: {movie['runtime']} min" if movie.get("runtime") else "Runtime: unknown",
        f"Rating: {movie.get('vote_average')}",
        f"Countries: {', '.join(movie.get('country') or []) or 'unknown'}",
        f"Original language: {movie.get('original_language') or 'unknown'}",
        f"Overview: {movie.get('description') or 'not available'}",
    ]
    return "Selected movie (full catalog facts):\n" + "\n".join(lines)


async def astream_conversational(
    message: str, history: list[dict] | None = None, system_prompt: str = CONVERSATIONAL_SYSTEM_PROMPT,
    selected_movie: dict | None = None, selected_position: int | None = None,
):
    """Stream a conversational response (no movie search). Yields tokens.

    `selected_movie` is the catalog row of the card the user is viewing; its
    facts ride on the current turn so "this"/"it" questions are answered from it;
    `selected_position` is its 1-based place in the latest list.
    """
    history = history or []
    current_content = message
    if selected_movie:
        current_content += f"\n\n{_selected_movie_block(selected_movie, selected_position)}"
    messages = _build_chat_messages(
        history=history,
        system_prompt=system_prompt,
        current_content=current_content,
        max_turns=settings.SESSION_HISTORY_MAX_TURNS,
        max_chars=settings.SESSION_HISTORY_MAX_CHARS_PER_ENTRY,
    )
    try:
        with inference_operation("llm", "conversational"):
            think_filter = _ThinkStreamFilter()
            async for content in get_inference_backends().llm.astream(
                **_stream_chat_payload(messages),
                timeout=settings.LLM_TIMEOUTS["explanation"], call_type="conversational",
            ):
                visible = think_filter.feed(content)
                if visible:
                    yield visible
            tail = think_filter.flush()
            if tail:
                yield tail
    except (httpx.HTTPError, InferenceError, json.JSONDecodeError) as e:
        record_failure("llm", "conversational", e)


async def astream_explanation(
    query: str, movies: list[dict], history: list[dict] | None = None, unverified: list[str] | None = None,
):
    """Async streaming variant of generate_explanation. Yields tokens as they arrive.

    `unverified` lists conditions the user asked for that the catalog cannot
    check (dropped with their agreement); the turn lists them in a
    "[Not checked: ...]" field that the system prompt explains."""
    history = history or []
    unverified_line = EXPLANATION_UNVERIFIED_LINE.format(
        conditions=", ".join(f'"{c}"' for c in unverified)
    ) if unverified else ""
    messages = _build_chat_messages(
        history=history,
        system_prompt=EXPLANATION_SYSTEM_PROMPT,
        current_content=EXPLANATION_TURN_TEMPLATE.format(
            query=query, movies_context=_build_movies_context(movies), unverified_line=unverified_line,
        ),
        max_turns=settings.SESSION_HISTORY_MAX_TURNS,
        max_chars=settings.SESSION_HISTORY_MAX_CHARS_PER_ENTRY,
    )
    try:
        with inference_operation("llm", "explanation"):
            think_filter = _ThinkStreamFilter()
            async for content in get_inference_backends().llm.astream(
                **_stream_chat_payload(messages),
                timeout=settings.LLM_TIMEOUTS["explanation"], call_type="explanation",
            ):
                visible = think_filter.feed(content)
                if visible:
                    yield visible
            tail = think_filter.flush()
            if tail:
                yield tail
    except (httpx.HTTPError, InferenceError, json.JSONDecodeError) as e:
        record_failure("llm", "explanation", e)


async def astream_explanation_parts(
    query: str, movies: list[dict], history: list[dict] | None = None, unverified: list[str] | None = None,
):
    """Stream the explanation already routed into slots. Yields `(slot, text)`.
    `history` and `unverified` are forwarded to `astream_explanation`.

    `slot` is None for the opening line and a 0-based index into `movies` for
    each per-movie description -- see `ExplanationSplitter`. `astream_explanation`
    is deliberately left as a raw token generator so the splitting stays
    testable on its own, without an HTTP stream in the way.
    """
    splitter = ExplanationSplitter(len(movies))
    async for token in astream_explanation(query, movies, history, unverified):
        for part in splitter.feed(token):
            yield part
    for part in splitter.flush():
        yield part


async def ais_available() -> bool:
    """Async variant of is_available."""
    return await get_inference_backends().llm.ais_available()


def _fallback_intent(_query: str) -> dict:
    """Return empty intent when the configured LLM is unavailable.

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
