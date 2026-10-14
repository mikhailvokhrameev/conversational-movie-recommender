# 0004. User constraints are SQL hard filters

Status: Accepted

## Context

Users state constraints: "not horror", "nothing from the US", "rated above 7", "after
2010", "under two hours", "in Korean". Treating them as one more scoring signal
(for a negation, `metadata = 0.0`) still lets a forbidden movie through whenever its
semantic similarity is high. Asking the LLM to check each result is slow, sampled and
can be argued out of the constraint.

## Decision

The LLM only extracts the constraint into the intent; code enforces it. In
`candidate_generation._apply_hard_filters` they become `WHERE` / `exclude` clauses
on the catalog before any ranking:

- negated genres, excluded and required countries;
- `min_vote_average`, release-year range, runtime range;
- `original_languages`.

A movie with unknown rating, release date or runtime passes (unknown means "cannot
judge", not "fails"); `original_languages` has no such exception because the language
is always known for an imported row. Popularity is the deliberate opposite: TMDB's
metric is unbounded and non-linear, so no threshold could be picked reliably and it
stays a scoring signal ([0003](0003-weighted-scoring-rerank-blend-mmr.md)).

## Consequences

- A violation cannot reach the user through a ranking quirk. `evaluate_scoring` reports
  `negation_violations` as a safety metric separate from relevance.
- Filters apply before the vector ordering, which is why exact search is used
  ([0002](0002-hybrid-retrieval-exact-knn-lexical-rrf.md)).
- A wrongly parsed constraint is enforced wrongly and silently excludes good movies.
  Parsing quality is therefore evaluated end to end
  ([0007](0007-evaluation-methodology.md)).
- Stacked filters can empty the pool; the reply then relies on what survives, or on the
  relevance gate once it is calibrated ([0009](0009-relevance-gate-off-until-calibrated.md)).

Details: [ML.md, Candidate Generation](../ML.md#candidate-generation) ("Negation as hard
filter"), [core.md, Design Decisions](../core.md#design-decisions).
