# 0006. `params.yaml` is the single source of truth

Status: Accepted

## Context

A recommender has many numbers that interact: LLM and embedder names, RRF constants,
four scoring weights, the blend weight, thresholds, token budgets, caps. Scattered
across code, env vars and Docker files they drift apart, and a result cannot be tied
to the settings that produced it. Some values are coupled -- `session.history_max_turns`
must fit inside `llm.num_ctx`, which must fit the VRAM budget -- and only make sense
documented next to each other.

## Decision

Every model name, weight, threshold, size and timeout lives in `params.yaml` at the
repo root. It is read once by `recommender/settings.py`, which exposes the values as
Django settings and validates them at load; the ollama container's entrypoint reads
`llm.model` from the same file. Environment variables are for secrets and deployment
wiring only (`SECRET_KEY`, `DB_PASSWORD`, `DEBUG`, ...), and nothing in the file is
overridable by one. Each value carries a comment saying why it is what it is, and a new tunable
goes into the file instead of into code (`CLAUDE.md` makes that a rule).

Load-time validation refuses an invalid file instead of running on it: scoring weights
must be non-negative and sum to 1.0, `reranking.weight` in [0, 1],
`reranking.min_score` strictly inside (0, 1) or null, `llm.num_ctx` and the history
caps at least 1.

## Consequences

- One diff shows what changed between two runs. Evaluation run records store the
  sha256 of `params.yaml` ([0007](0007-evaluation-methodology.md)).
- Some changes are not free despite being one line: `embedding.model`,
  `embedding.dimensions`, `embedding.text.*` and `embedding.revision` invalidate the
  stored vectors ([0008](0008-embedding-index-provenance.md)); a different
  `llm.num_ctx` reloads the model and changes VRAM.
- Pins live here too: Hugging Face `revision`s for the embedder and reranker. The
  Ollama image tag is pinned in `ollama/Dockerfile`, and the LLM tag cannot be pinned
  by digest (the served digest is logged instead).
- Changing a value needs a restart; there is no runtime override by design.

Details: [infrastructure.md, Configuration](../infrastructure.md#configuration),
[Reproducibility](../infrastructure.md#reproducibility), [params.yaml](../../params.yaml).
