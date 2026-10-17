"""Session state helpers: explicit-preference tracking and per-movie history facts.

A session stores no embedding. Each turn's retrieval is driven by the query
itself (the LLM's `semantic_query` plus, in the lexical leg, the user's message).
"""


def track_explicit_preferences(
    current_preferences: dict,
    intent: dict,
) -> dict:
    INTENT_TO_PREF = [
        ("genres", "liked_genres"),
        ("negations", "disliked_genres"),
        ("themes", "themes"),
        ("reference_films", "reference_films"),
    ]

    prefs = {pref_key: list(current_preferences.get(pref_key, [])) for _, pref_key in INTENT_TO_PREF}

    for intent_key, pref_key in INTENT_TO_PREF:
        for item in intent.get(intent_key, []):
            if item not in prefs[pref_key]:
                prefs[pref_key].append(item)

    return prefs


def build_movie_facts(movies: list[dict], description_chars: int) -> list[dict]:
    """Per-movie facts stored with an assistant recommendation turn, in display order.

    This is what later follow-ups ("which one is shorter?") are answered from:
    the catalog holds no director/cast, so without these the model falls back
    to parametric memory. The description is cut here, at storage time, so the
    stored history -- not just the replay -- stays bounded by the same budget.
    `year` comes from `release_date` (ISO string or date), None if missing.
    """
    facts = []
    for movie in movies:
        release_date = movie.get("release_date")
        year = int(str(release_date)[:4]) if release_date else None
        facts.append({
            "title": movie["serial_name"],
            "year": year,
            "genres": list(movie.get("genres") or []),
            "runtime": movie.get("runtime"),
            "description": (movie.get("description") or "")[:description_chars],
        })
    return facts
