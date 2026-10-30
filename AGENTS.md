# AGENTS.md

Conversational movie recommender: Django/uvicorn + Postgres/pgvector backend, React/Vite frontend, local or OpenRouter inference (English chat over the TMDB catalog, hybrid semantic+lexical retrieval, cross-encoder reranked).

**IMPORTANT: never change `embedding.model`/`embedding.dimensions` without a schema migration + `generate_embeddings --all` — old vectors aren't convertible. The `EmbeddingIndex` row records what built the vectors (model, dimensions, text template version, `embedding.text.*`); `check_embedding_index` in the start-up chain stops the container on a mismatch. Bump `EMBEDDING_TEXT_VERSION` when the text template changes.**
The embedder/reranker load at the pinned Hugging Face commits `embedding.revision` / `reranking.revision` (null = unpinned); moving the embedding one also needs `generate_embeddings --all`, but a null on either side of the comparison is only a warning.

## Commands
- Run all: `docker compose up --build`
- Backend tests: `docker compose exec backend pytest -v` (single: add `::test_name`)
- Backend deps (uv, `backend/uv.lock`): `cd backend && uv sync` / `uv run pytest`; after editing `pyproject.toml` run `uv lock` and commit the lock
- Frontend: `cd frontend && npm run dev` / `npm run lint` / `npm run build`
- Embeddings: `docker compose exec backend python manage.py generate_embeddings`
- Eval retrieval: `docker compose exec backend python manage.py evaluate_scoring --sweep`
- Calibrate the relevance gate (golden hits): `docker compose exec backend python manage.py evaluate_scoring --intent oracle --gate-report`
- Export dev results for hand-labelling: `docker compose exec backend python manage.py evaluate_scoring --intent oracle --k 10 --label-export eval_runs/labels.csv` (lands in `./eval_runs`, the container's writable mount)
- Gate curves from the labelled CSV (no DB): `cd backend && python manage.py evaluate_scoring --label-report ../eval_runs/labels.csv`
- Is the session vector worth keeping: `docker compose exec backend python manage.py evaluate_scoring --session-ablation`

## Anti-patterns
- No new env vars/hardcoded model or weight values — everything tunable is in `params.yaml`.
- Don't treat `evaluate_scoring --llm-judge` as ground truth — same model family as the system, so it's circular (see `TODOS.md`).

## Caveats
- `params.yaml` is the single source of truth for models/weights (LLM, embedder, RRF, reranker, EMA). Env vars are secrets/wiring only.
- LLM emits `<think>` blocks; suppressed via `params.yaml` (`llm.thinking: false`) *and* stripped defensively in `llm_client.py` — keep both, JSON parsing needs bare output.
- Local mode targets NVIDIA GTX 1080 Ti (11GB); Qwen3.5 9B with context 6144 still requires user VRAM/compatibility validation. OpenRouter mode needs no GPU or local model load; see `docs/infrastructure.md`.
- Retrieval: exact cosine (no HNSW), semantic+lexical fused via RRF, then reranked (`bge-reranker-v2-m3`); reranker `top_k` is the latency dial.
- Postgres on host port **5433**, not 5432.

## Docs
`docs/infrastructure.md`, `docs/ML.md`, `docs/core.md`, `docs/api.md`, `docs/models.md`, `docs/frontend.md`, `docs/adr/README.md` (decision records), `docs/datasheet.md` (movie catalog), `params.yaml`, `TODOS.md`

## Inference modes and agent boundaries
- **Agents must NEVER perform real API calls**, including research requests, preflight, application startup, model downloads, or integration checks through commands, Docker, browser, SDKs or subprocesses. HTTP tests must use mocks; host tests must block external networking. Only the user runs real API/GPU checks. Do not read or edit the user's `.env`.
- Keep one recommendation pipeline: shared prompts, validation, retrieval, scoring, normalization, blending, gate, MMR, explanations, history, splitter and frontend SSE. Select providers only in the cached adapter factory; no mode branches in views/pipeline/scoring.
- `INFERENCE_MODE=local` is the default; `openrouter` explicitly selects the cloud profile in `params.yaml`. No automatic provider fallback. Restart after configuration changes.
- Local launch: `docker compose up --build`. Cloud launch: `docker compose -f docker-compose.yml -f docker-compose.openrouter.yml up --build` (excludes automatic local override). Set `OPENROUTER_API_KEY` yourself using `.env.example`.
- Before switching, use the current mode's Compose files with `down` **without `-v`**; preserve volumes. Startup never rebuilds embeddings. Origin changes trigger a deterministic read-only sample compatibility check which may make real embedding requests; only the user runs it. Passing permits reads only, never mixed-origin partial generation. On failure restore original mode or explicitly rebuild an approved separate catalog; never auto-run `--all`.
- Provenance includes canonical model, dimensions, text template/settings, revision, provider and API provider. Cloud API model ID is separate; unknown API revision stays null. Keep dimensions 1024 and pinned local revisions.
- `check_inference` sends real small requests through all three active adapters (potential API costs); user only. Full cloud embedding lifecycle must first use a separate test database/project; separately read-check the existing local index without rebuilding it.
- Real GPU/API, latency and quality acceptance remains pending until user evidence. Record Ollama version/digest, GTX 1080 Ti VRAM/offload, context 6144, long-dialogue TTFT/latency, JSON and complete explanations. Insufficient VRAM is blocking; do not shrink the agreed model/context automatically.
- Preserve recommendation-quality tasks 5–8 and evaluation flags; Voyage's null gate is uncalibrated. Compare separate dev run records on the same data and user labels; the LLM judge is never ground truth.
