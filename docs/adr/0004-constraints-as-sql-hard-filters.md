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

A movie with an unknown rating passes `min_vote_average` (the catalog keeps only
movies with `catalog.min_vote_count` votes, so rating is always known in practice).
`original_languages` has no such exception either, since the language is always known
for an imported row.

> **Superseded in part (user decision 2026-10-17).** The original rule let a movie with
> an unknown release date or runtime pass the year and runtime filters ("unknown means
> cannot judge, not fails"). When the user sets a year or runtime bound, a movie with
> an unknown value now does **not** pass: "from 2026" returned a film with an empty
> release date. TMDB writes `runtime 0` for unknown, so the importer stores it as NULL
> and migration 0009 nulls existing zeros, which keeps them from satisfying "under 20
> minutes". Popularity is the deliberate opposite: TMDB's
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
