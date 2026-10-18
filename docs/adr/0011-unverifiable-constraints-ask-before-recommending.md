# 0011. Ask before dropping constraints the catalog cannot check

Status: Accepted

## Context

Manual testing on the GPU machine turned up answers that were not honest. "1950s movies
with Keanu Reeves" returned five 1950s films, as if the actor had been checked. After a
set of short animated documentaries, "They are not silent" returned the same five films
again. The catalog has no cast, director, silent or black-and-white, awards or studio
data, and `MessageIntent` had no way to say that a condition was ignored. Refinements also
repeated earlier results, because nothing excluded movies already shown.

When the hard filters left nothing, the reply was a generic "nothing matched, rephrase",
which does not tell the user which constraint to loosen.

## Decision

1. `MessageIntent.unsupported_constraints: list[str]` holds the conditions the catalog
   cannot check, in the user's words. The classify+parse prompt lists what the catalog
   can check (genre, mood/theme, country, rating, release year, runtime, language) and
   what it cannot. It includes the examples "1950s movies with Keanu Reeves" and "They
   are not silent", and a rule: once the assistant has offered to search without a
   condition and the user agreed, the condition is not listed again. The field is not a
   filter. It is in `NON_FILTER_FIELDS`, so it is left out of `INTENT_SCHEMA` and out of
   the intent dict that reaches the pipeline and the client. `standalone_query` leaves
   these conditions out too, so the reranker does not rank on "Keanu Reeves".
2. If the list is not empty, `ChatView` does not search. It sends a fixed reply without
   an LLM call ("I can't check "with Keanu Reeves" -- the catalog has no cast, director,
   ... data. Want me to search without that?") and `movies: []`. It stores the assistant
   turn with `pending_relaxation: {"dropped": [...]}`. On the next turn, constraints in
   `dropped` are ignored (case-insensitive) even if the classifier lists them again, so
   a "yes" searches. A new unsupported condition still gets asked about. A "no" is an
   ordinary next turn.
3. An empty result (hard filters, or later the relevance gate) gets a fixed reply that
   names the active hard filters, built from the intent dict: "Nothing in the catalog
   matches all of: released 1950–1959, under 20 min, rating ≥ 9.0. Which one should I
   loosen?". Genres, mood and themes only rank and never empty a result, so they are not
   listed. With no hard filter active the old generic message is sent.
4. Each assistant turn that showed movies stores their `tmdb_id`s as `movie_ids`. A
   `refinement` passes every `movie_id` shown within the replayed history window
   (`session.history_max_turns` pairs) as `exclude_ids` to `generate_and_score` (the
   mechanism from the reference-film exclusion). A `new_search` excludes nothing.

## Consequences

- An honest "can't check that" or "nothing matches" never names a movie: both replies
  are templates, not LLM output.
- The routing depends on the classifier filling `unsupported_constraints`. A miss falls
  back to the old behaviour (the condition is silently ignored). A false positive costs
  one extra question.
- Excluding every movie shown in the window shrinks the pool on long refinement chains.
  Stored history is not capped, so the exclusion uses the same window as the LLM replay:
  at most `history_max_turns` sets of `top_n` movies, and only those the model still
  sees. A movie shown earlier than that can come back.
- `evaluate_scoring` in e2e still searches a query with `unsupported_constraints`, counts
  it in the new `unsupported_constraints` counter and flags its row
  (`UNSUPPORTED(...)`): in production that query would show no movies, so its nDCG is
  not what a user would get.
- The golden set can express "the right answer is a question" (`expected: "ask"`,
  correct iff the e2e parse fills `unsupported_constraints`) and "the right answer is
  nothing" (`expected: "empty"`), and prior turns can include assistant replies
  (`history`) -- see [0007](0007-evaluation-methodology.md). Only the text is replayed,
  though: the exclusion of shown movies and the `pending_relaxation` follow-up are not
  reproduced in evaluation.
- Loading cast and crew from TMDB would make some of these constraints checkable. That
  is a separate project.
