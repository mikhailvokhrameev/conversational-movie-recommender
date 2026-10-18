# 0007. Evaluation: golden set, e2e vs oracle, dev/test split, judge is not ground truth

Status: Accepted (the machinery); no results are recorded yet

## Context

The first evaluation was misleading in ways that are easy to repeat. Relevance was
"shares a genre with the query's labelled genres", which is what the `metadata` scoring
signal computes from the same genres, so precision was inflated by construction. It
bypassed the production retrieval path, so eval and production could drift. Weights
were tuned and reported on the same queries. The only independent check was an LLM
judge from the same model family as the system, which shares its blind spots, and a
failed judge call was silently scored 3.

## Decision

`manage.py evaluate_scoring` scores against a labelled golden set
(`data/golden_set.json`):

- **Relevance is golden-set membership** (`{title, year}` or `tmdb_id` references,
  graded 1 or 2), resolved against the catalog at run time. Unresolved references are
  listed, and a query with none resolved is reported as unscorable, not zero.
- **Two intent modes.** `e2e` (default) parses with the production call
  `aclassify_and_parse`, so parsing is evaluated too. `oracle` uses each entry's labelled
  intent and, for multi-turn entries, its hand-written standalone query (so follow-up
  fragments are not ranked as-is), isolating retrieval. The gap between them is the cost of parsing errors.
- **The production path.** Retrieval is `core.pipeline.generate_and_score`, the same
  function the chat view runs. Multi-turn entries are parsed with their prior
  messages as history, as production does.
- **Three kinds of right answer.** An entry's `expected` is `results` (default: a
  ranked list, scored by golden membership), `empty` (the catalog cannot serve it:
  correct iff nothing is returned) or `ask` (a condition the catalog cannot check:
  correct iff the e2e parse lists it in `unsupported_constraints`, which makes the
  chat ask, [0011](0011-unverifiable-constraints-ask-before-recommending.md)).
  `empty`/`ask` entries carry no `relevant` list, are counted (`expected_*` counters,
  per-query `outcome_correct`) rather than accuracy-scored, and stay out of the
  aggregates and the sweep. Oracle mode skips `ask` entries (no parse) and counts them.
  Prior turns can be written as `history` with assistant replies; only their text is
  replayed, not the session state production keeps beside them.
- **Dev/test split.** `--sweep` and `--gate-report` read only `dev` and refuse
  `test`/`all`; the default eval reports `test`.
- **The LLM judge is a secondary signal.** `--llm-judge` is reported beside the
  golden-set metrics, never used to rank configurations, and failures are counted
  (`judge_failures`), not averaged in. It is circular and is not ground truth.
- **Reproducible and recorded.** Structured calls run at temperature 0 with a fixed
  seed; every run writes a JSON record with the commit, `params.yaml` hash, golden-set
  hash, options, per-query results and counters.

## Consequences

- The labels are a **draft** written from film knowledge without catalog access; they
  need human review before any number means much (see `TODOS.md`, "Evaluation").
  ~30 queries is thin, and lists are samples, so precision is a lower bound.
- No evaluation has been run on the current system; `ML.md` "Evaluation Results" says
  so rather than carrying old numbers.
- The golden set prices both sides of a relevance gate: its cost on answerable queries
  and, through the `expected: "empty"` entries, its benefit
  ([0009](0009-relevance-gate-off-until-calibrated.md)). Five such entries and three
  `ask` entries are a start, not a measurement; an `empty` label is itself a claim
  about the catalog that needs the same human review as the rest.
- Session state (`pending_relaxation`, the movies a refinement excludes) is not
  replayed, so a `yes` after an ask, or a refinement's exclusion, is not evaluated.
- A stronger or different-family judge is the listed way out of the circularity; it is
  not implemented.

Details: [ML.md, Evaluation Framework](../ML.md#evaluation-framework),
[core.md, evaluation.py / golden_set.py](../core.md#modules).
