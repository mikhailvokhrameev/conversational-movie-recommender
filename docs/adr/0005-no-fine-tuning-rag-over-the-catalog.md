# 0005. No fine-tuning: RAG over the catalog

Status: Accepted

## Context

Two things would need domain knowledge: judging which movies fit a request, and
talking about specific movies. Fine-tuning the LLM or the embedder is the other way to
supply it. Against that: there is no interaction data (no accounts, watch history or
ratings -- see "Production Extensions" in ML.md), the only labelled data is a
~30-query draft golden set, and every swap of a fine-tuned model would need its
own evaluation on that thin set. The product is a local, single-GPU (~11 GB) project.

## Decision

Use general pretrained models as shipped and put the catalog knowledge in at query
time:

- **Embedder** `BAAI/bge-m3`, not fine-tuned. Movies are embedded as title, genres,
  a description prefix and keywords; the vectors live in pgvector.
- **LLM** plain `qwen3:8b` from Ollama's own library, not a language- or
  domain-adapted variant. Explanations and follow-up answers get movie metadata and
  descriptions in the prompt (`[[n]]` movie list, `movie_facts` from the latest shown
  set), and the conversational prompt says to admit a detail is not in the catalog
  rather than state it.
- **Reranker** `BAAI/bge-reranker-v2-m3`, not fine-tuned.

All three are pinned instead of trained ([0006](0006-params-yaml-single-source-of-truth.md),
[0008](0008-embedding-index-provenance.md)).

## Consequences

- The catalog can be re-imported and re-embedded without retraining anything; only the
  embedding step is repeated, and the provenance check enforces it.
- Quality is capped by what general models know about movies; facts outside the catalog
  (cast, director, awards -- not imported) cannot be grounded and the model may fall back
  on memory unless it follows the prompt.
- If labelled data appears (reviewed golden set, real feedback), fine-tuning becomes
  testable. Not before: there would be nothing honest to measure it against
  ([0007](0007-evaluation-methodology.md)).

Details: [ML.md, Embedding Model](../ML.md#embedding-model) ("No fine-tuning needed"),
[ML.md, LLM](../ML.md#llm-ollama), [core.md, ollama_client.py](../core.md#modules).
