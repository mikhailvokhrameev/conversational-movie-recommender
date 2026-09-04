"""Maintenance of the Movie.search_vector full-text column.

The lexical channel exists to catch what embeddings miss: exact entities.
A query naming a specific title or actor ("что-то как Место встречи
изменить нельзя", "фильмы с Хабенским") needs literal token matching --
embeddings encode meaning, not names.

Only entity-bearing fields are indexed:

  serial_name  weight A   the title itself, the strongest exact-match signal
  director     weight B   a name
  actors       weight B   names (jsonb array, flattened to text)

`description` is deliberately excluded. It is long free text, so it would
dominate the index by sheer token count and match on ordinary vocabulary --
and it is precisely what the semantic channel already handles well. Keeping
it out preserves the split: lexical finds names, semantic finds meaning.

Text search config is 'russian' (snowball stemmer + stopwords). Russian is
heavily inflected, so stemming is what lets a query for "Место встречи"
match the catalogued "Место встречи изменить нельзя".
"""

from django.db import connection

# One canonical definition of the vector, used by both the backfill migration
# and post-import refreshes so the two can never drift apart.
_SEARCH_VECTOR_EXPR = """
    setweight(to_tsvector('russian', coalesce(serial_name, '')), 'A')
    || setweight(to_tsvector('russian', coalesce(director, '')), 'B')
    || setweight(to_tsvector('russian', coalesce(
           CASE WHEN jsonb_typeof(actors) = 'array'
                THEN (SELECT string_agg(a.value, ' ')
                      FROM jsonb_array_elements_text(actors) AS a(value))
                ELSE '' END, '')), 'B')
"""


def refresh_search_vectors(only_missing: bool = False) -> int:
    """Recompute search_vector for catalog rows. Returns rows updated.

    Set only_missing=True after an incremental import to touch just the new
    rows; leave it False to rebuild everything (e.g. after changing the
    vector definition above).
    """
    sql = f"UPDATE movies_movie SET search_vector = {_SEARCH_VECTOR_EXPR}"
    if only_missing:
        sql += " WHERE search_vector IS NULL"

    with connection.cursor() as cursor:
        cursor.execute(sql)
        return cursor.rowcount
