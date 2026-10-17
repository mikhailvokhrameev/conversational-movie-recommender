# 0002. Hybrid retrieval: exact cosine kNN, a lexical channel, RRF fusion

Status: Accepted

## Context

Candidate generation has three open questions. (1) Which nearest-neighbour index.
The first framing was "HNSW or IVFFlat", which skipped whether approximate search
is needed at all. (2) Embeddings encode meaning, not identity: "something like
Interstellar" can miss the exact record because nothing in the vector space favours
the literal name. (3) A cosine distance and a `ts_rank` are not comparable numbers.

## Decision

- **Exact cosine kNN, no ANN index.** At ~50K x 1024 float32 vectors a sequential
  scan is tens of milliseconds in Postgres, under 2% of a request that already
  spends ~2 s in the LLM. Hard filters ([0004](0004-constraints-as-sql-hard-filters.md))
  are applied as `WHERE` before the vector ordering. An HNSW scan walks its graph and
  post-filters, so a selective filter can return far fewer than the 100 requested
  candidates; exact search ranks only what truly qualifies.
- **A lexical channel** -- Postgres `tsvector` over title, original title (weight A)
  and keywords (weight B), `english` config, OR-ed terms. Descriptions are left out on
  purpose: the semantic channel already handles them and they would swamp the index.
- **Fusion by Reciprocal Rank Fusion** (`weight / (k + rank)`, k = 60), semantic 1.0,
  lexical 0.7, so lexical loses ties instead of being suppressed. RRF uses rank only,
  which is the only trustworthy thing across independent systems.

## Consequences

- Recall is exact and filters never starve the pool.
- Cost grows linearly with the catalog. Revisit at roughly 10x (~500K+ items) and
  re-measure before reintroducing an index.
- Fusion decides which candidates enter the pool, not their final order: the scorer has
  no lexical term, so a pure title match would rank low without the cross-encoder
  ([0003](0003-weighted-scoring-rerank-blend-mmr.md)).
- The lexical channel depends on the English catalog and the `english` config
  (`retrieval.lexical` in `params.yaml`).

Details: [ML.md, Candidate Generation](../ML.md#candidate-generation),
[ML.md, Lexical Retrieval Channel](../ML.md#lexical-retrieval-channel).
