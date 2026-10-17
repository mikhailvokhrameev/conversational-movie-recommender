# 0010. Drop the session vector, rank on a context-aware query text

Status: Accepted

## Context

Two findings from one GPU session.

**The session vector did not earn its upkeep.** `evaluate_scoring --session-ablation`
(e2e intent, 18 multi-turn golden entries) scored each entry with and without the
EMA session vector. The paired nDCG@5 difference, with minus without, was
-0.0143 on average (95% bootstrap CI [-0.0310, +0.0005]); the vector helped on 1
query, changed nothing on 3 and hurt on 14. Verdict: `REMOVE`. Without the vector
the multi-turn nDCG@5 is 0.073. The vector also cost a pgvector column tied to
`embedding.dimensions`, an EMA with two alphas and a recompute under a row lock.

**Lexical search and the reranker saw only the raw last message.** `ChatView` passed
`query_text=message` to `generate_candidates` and `rerank_candidates`. After "I liked
The Godfather", the message "suggest something similar" has no content of its own: the
cross-encoder scored "Something Like Happiness" 0.17 and The Godfather 0.0004, and the
lexical channel matched on "something" and "similar". The LLM had already resolved the
context into `semantic_query` and the filters, but the two components that read text
never saw it.

## Decision

1. Remove the `session` scoring signal, the stored vector and its EMA
   (`ChatSession.preference_vector`, `session.alpha`, `session.alpha_refinement`,
   `session_manager.update_preference_vector` / `vector_for_scoring`) and the
   `--session-ablation` mode with its `evaluation.session_ablation` settings.
2. `aclassify_and_parse` returns a `standalone_query`: the user's request rewritten so
   it makes sense without the conversation ("crime dramas similar to The Godfather").
   `ChatView` and `evaluate_scoring` pass it as `query_text` to lexical retrieval and
   the reranker, falling back to the raw message when it is missing. It is not sent to
   the client in the `intent` event.
3. Lexical search drops conversational stopwords (`retrieval.lexical.stopwords` in
   `params.yaml`: "movie", "like", "show", "something", ...) before the max-terms cut.
   If no term is left, the lexical channel is skipped and semantic search carries the
   query.

## Consequences

- Scoring weights are renormalized: semantic 0.4667, metadata 0.3333, popularity 0.2
  (were 0.35 / 0.25 / 0.25 session / 0.15 popularity in
  [0003](0003-weighted-scoring-rerank-blend-mmr.md)). Semantic and metadata keep their
  old 7:5 proportion and share the remaining 0.8. `evaluate_scoring --weights` takes
  `semantic,metadata` and scales the pair to `1 - popularity`.
- Migration 0008 drops `ChatSession.preference_vector`. The data is lost, which costs
  nothing: sessions expire after 24 h and nothing else read the column. Changing
  `embedding.dimensions` now concerns only `Movie.embedding`.
- `--intent oracle` still ranks on the raw last message: oracle entries carry a labelled
  intent but no `standalone_query`, and the mode exists to isolate retrieval. Oracle
  runs on multi-turn entries therefore do not exercise the contextual query text; use
  `e2e` to measure it.
- The query text now depends on one more LLM field. An empty
  `standalone_query` is replaced by the raw message in `aclassify_and_parse`, which degrades to the old behaviour (raw message), not to an error.
- Topic switches are no longer biased toward the previous turns by a stored vector;
  context reaches ranking only through the LLM's parse of the history.
