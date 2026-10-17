# Architecture Decision Records

Short records of the decisions that shape the recommender: what was decided, why,
and what it costs. They are an index into the detailed reasoning, which stays in
[ML.md](../ML.md), [core.md](../core.md) and the comments of
[params.yaml](../../params.yaml); each record links to the sections it summarises
rather than copying them.

Format: Title, Status, Context, Decision, Consequences. Status is `Accepted`
(in the code and in use) or `Proposed` (decided in principle, the code says a
piece is still pending). A decision that is reversed gets a new record that
supersedes the old one; the old one is kept with status `Superseded by NNNN`.

| # | Decision | Status |
|---|----------|--------|
| [0001](0001-llm-parses-and-explains-never-ranks.md) | The LLM parses and explains, it never ranks | Accepted |
| [0002](0002-hybrid-retrieval-exact-knn-lexical-rrf.md) | Hybrid retrieval: exact cosine kNN, a lexical channel, RRF fusion | Accepted |
| [0003](0003-weighted-scoring-rerank-blend-mmr.md) | Min-max weighted scoring, cross-encoder blend, MMR | Accepted |
| [0004](0004-constraints-as-sql-hard-filters.md) | User constraints are SQL hard filters | Accepted |
| [0005](0005-no-fine-tuning-rag-over-the-catalog.md) | No fine-tuning: RAG over the catalog | Accepted |
| [0006](0006-params-yaml-single-source-of-truth.md) | `params.yaml` is the single source of truth | Accepted |
| [0007](0007-evaluation-methodology.md) | Evaluation: golden set, e2e vs oracle, dev/test split, judge is not ground truth | Accepted |
| [0008](0008-embedding-index-provenance.md) | The embedding index records what built it, and start-up checks it | Accepted |
| [0009](0009-relevance-gate-off-until-calibrated.md) | The relevance gate ships off until it is calibrated | Proposed |
| [0010](0010-drop-session-vector-contextual-query-text.md) | Drop the session vector; rank on a context-aware standalone query | Accepted |
| [0011](0011-unverifiable-constraints-ask-before-recommending.md) | Ask before dropping constraints the catalog cannot check | Accepted |

To add one: copy the shape of an existing record, take the next number, add a row
above.
