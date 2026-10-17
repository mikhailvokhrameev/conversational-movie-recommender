# 0008. The embedding index records what built it, and start-up checks it

Status: Accepted

## Context

Stored vectors are only comparable with query vectors from the same model, text
template and (for pinned weights) revision. A change to any of them leaves vectors of the
same width in place: nothing raises, similarities are just meaningless, and
recommendations degrade silently. `embedding.dimensions` is baked into the pgvector
columns, but the model name, the text template and the revision are not.

## Decision

A single-row `EmbeddingIndex` table records `embedding.model`, `embedding.dimensions`,
`embedding.revision`, the text template version (`EMBEDDING_TEXT_VERSION`) and
`embedding.text.*`. `generate_embeddings` writes it. `check_embedding_index` runs in the
backend start-up chain (`docker-compose.yml`, after `import_catalog`) and compares the
record with `params.yaml`:

- a mismatch stops the container, with the fix: revert the config or re-embed;
- no record and no embeddings is a notice; embeddings without a record ("legacy") is a
  warning;
- the revision is compared only when both sides are set (a null is a warning, not a
  mismatch), so an index recorded before pinning keeps working.

`generate_embeddings` refuses to embed only the missing rows under a mismatch (that
mixes vector spaces). `--all` re-embeds everything and refreshes the record; `--adopt`
records the current config for a legacy index without re-embedding, as an explicit
assertion by the operator. The embedded text is built by a versioned function
(`build_movie_embedding_text`); any template change bumps `EMBEDDING_TEXT_VERSION`.

## Consequences

- A forgotten re-embed becomes a loud start-up failure instead of silent degradation.
- Changing the embedder is a deliberate procedure: schema migration if the dimensions
  change, then `generate_embeddings --all`. `CLAUDE.md` states it.
- The reranker has no stored artifacts, so its revision pin is not recorded in the DB.
- `--adopt` trusts the operator: it cannot verify the vectors were built with the
  current config.
- Postgres-gated tests cover the commands; they could not run on the build host.

Details: [ML.md, Embedding Model](../ML.md#embedding-model) ("Index provenance"),
[models.md, EmbeddingIndex](../models.md#embeddingindex),
[infrastructure.md](../infrastructure.md#reproducibility).
