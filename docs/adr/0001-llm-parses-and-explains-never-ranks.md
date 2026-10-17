# 0001. The LLM parses and explains, it never ranks

Status: Accepted

## Context

The system could be a standard RAG chat: retrieve documents, hand them to the LLM,
let it decide what is relevant and say so. For a recommender that makes the
ranking opaque, non-repeatable (sampling) and impossible to tune or test
without reading generated text. The LLM here is a local 8B model (`qwen3:8b`),
which is the weakest part of the stack at judging a 50K-movie catalog and the
slowest (about 2 s per call).

## Decision

The LLM has two helper jobs and no say in the order of results:

1. **Input**: one `aclassify_and_parse` call returns the message category
   (`new_search`, `refinement`, `follow_up`, `general_chat`) plus a structured intent
   (genres, mood, negations, filters, a `semantic_query`).
2. **Output**: it writes the explanation for movies the deterministic pipeline
   already chose (`astream_explanation`).

Everything between those two calls -- candidate generation, scoring, reranking,
MMR -- is code with numeric inputs. The explanation is only diagnosed after the
fact: `check_explanation_titles` logs a retrieved-but-not-shown title that the text
mentions (it does not alter the streamed reply). If the LLM is down, `aclassify_and_parse` degrades to an empty
intent and semantic search still returns results.

## Consequences

- The ranking is explainable, unit-testable and tunable by weights
  ([0003](0003-weighted-scoring-rerank-blend-mmr.md)), and evaluable without a
  model in the loop for the `oracle` intent mode
  ([0007](0007-evaluation-methodology.md)).
- Parsing quality is still a ceiling: a wrong intent changes filters and the
  `semantic_query`. That is why evaluation has an `e2e` mode that includes the parse.
- Constraints the LLM extracts are enforced by SQL, not trusted to its judgement
  ([0004](0004-constraints-as-sql-hard-filters.md)).
- Structured replies are schema-constrained and validated, with a repair retry
  ([ML.md, LLM](../ML.md#llm-ollama)).

Details: [ML.md, System Architecture](../ML.md#system-architecture),
[core.md, Architecture](../core.md#architecture).
