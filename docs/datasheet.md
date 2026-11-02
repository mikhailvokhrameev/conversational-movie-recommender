# Datasheet: the movie catalog

Follows the headings of "Datasheets for Datasets", briefly. The catalog is what
the recommender searches; the golden set (last section) is what it is evaluated
against. Numbers marked *measured* were computed from the CSV on the maintainer's
machine by applying the import filter, not read from a running database.

## Motivation

The recommender needs a catalog of films with English descriptions, genres and
curated keywords, enough popularity/rating metadata to rank on, and a poster
per title for the UI. It was built to be searched in natural language, so the
text fields matter more than the numeric ones. It replaced an earlier ~18K-title
catalog from a different (Russian-language) source; the product is English now.
Nothing here was collected for this project: the data is a third-party export of
TMDB.

## Composition

One row per movie (`movies.Movie`, see [models.md](models.md)). Fields taken from
the CSV:

| Field | CSV column | Used for |
|-------|-----------|----------|
| `tmdb_id` | `id` | Identity (unique) |
| `serial_name`, `original_title` | `title`, `original_title` | Display, lexical search, embedded text |
| `genres` | `genres` | Metadata score, negation filter, embedded text |
| `country` | `production_countries` | Country include/exclude filters |
| `original_language` | `original_language` | Language filter |
| `keywords` | `keywords` | Lexical search, embedded text |
| `release_date` | `release_date` | Year filters, shown in answers |
| `description` | `overview` | Embedded text, explanations, reranker input |
| `runtime` | `runtime` | Runtime filters |
| `popularity`, `vote_average`, `vote_count` | same names | Popularity signal, rating floor |
| `poster_path` | `poster_path` | Poster image URL |

Not imported: `revenue`, `budget`, `backdrop_path`, `homepage`, `imdb_id`,
`tagline`, `production_companies`, `spoken_languages`. In particular there is **no
cast, director, age rating or awards data**; the assistant is told to say so rather
than answer from memory (see [core.md](core.md)).

Size: the raw file has 1,490,724 rows; **49,775** pass the import filter
(*measured*). Genres follow TMDB's 19-genre list; a title can have several.
Embeddings (1024-dim BGE-M3) and the full-text `search_vector` are derived columns,
not source data.

## Collection and source

The file is `TMDB Movie Dataset v11.csv` at the repo root (about 660 MB). It is
mounted read-only into the backend by `docker-compose.yml`
(`./TMDB Movie Dataset v11.csv:/app/TMDB Movie Dataset v11.csv:ro`), and
`settings.CATALOG_CSV_PATH` points at it. It is git-ignored, not stored in the
repository.

**Where the CSV was obtained, on what date, and under what licence is not recorded
in this repository.** The name suggests a community export of TMDB data, but that
is a guess. The maintainer must fill in: source URL, download date, version, and the
licence or terms the export was published under.

## Preprocessing and filters

`manage.py import_catalog` streams the CSV and keeps a row only if all of these hold
(`_passes_quality_filter`):

- `status` is `Released` (no unreleased titles);
- `adult` is not `True`;
- `overview` is non-empty after trimming;
- `poster_path` is non-empty;
- `vote_count >= catalog.min_vote_count` (20 in [params.yaml](../params.yaml)).

Kept rows are inserted in batches of `catalog.import_batch_size`; list-valued columns
(genres, countries, keywords) are split on `", "`; unparseable numbers fall back to
zero or null. Re-running with `--skip-existing` imports only unseen `tmdb_id`s. A
second step, `generate_embeddings`, embeds title, genres, the first 500 characters of
the description and up to 10 keywords, and records provenance
([ADR 0008](adr/0008-embedding-index-provenance.md)).

## Known limitations and biases

- **Popularity skew.** The vote-count floor removes the long tail of obscure titles by
  design, so the catalog over-represents what many people rated. The popularity
  scoring signal then pushes the same way. A film with under 20 votes cannot be
  recommended at all.
- **English and US-centric.** *Measured* on the filtered rows: 61% have English as
  original language and 49% list the United States among production countries; French,
  Italian, Spanish, Japanese and German follow far behind. Descriptions are English
  (`overview`), so non-English titles are retrieved through English summaries.
- **Recency.** About 46% of the filtered rows are from 2010 or later (*measured*).
- **Coverage ends 2023-09-30.** The CSV's `vote_count` snapshot froze around September
  2023, so films released from 2024 on have `vote_count` of at most 2 and fail
  `catalog.min_vote_count` (20). Lowering the threshold does not bring them back: only one
  2024 film reaches 5 votes (*measured*). The recommender therefore cannot offer films
  after that date. A "newest" request is counted back from the catalog's newest year
  (`intent.recent_years`, 2), and a year past the end gets a no-match reply that states
  where the catalog ends (`_no_match_message`, `core/catalog_bounds.py`).
- **TMDB's genre taxonomy.** Only 19 coarse genres, assigned by TMDB contributors, with
  no sub-genres or tone; mood and themes are therefore carried by description text and
  keywords, not by a taxonomy.
- **Sparse or inconsistent fields.** Keywords, runtime and countries are missing for
  some rows; the filters treat missing rating/date/runtime as "cannot judge" and keep
  the movie ([ADR 0004](adr/0004-constraints-as-sql-hard-filters.md)).
- **Crowd-sourced text.** Overviews and keywords are community-contributed on TMDB,
  so their length and quality vary from title to title.
- No fairness or demographic analysis was done; the data describes films, not people,
  but it inherits whose films TMDB's contributors list.

## Uses

Intended: natural-language movie recommendation and follow-up questions grounded in
title, year, genres, runtime and description; offline evaluation with the golden set.

Not suitable for: answering questions about cast, crew, awards or box office (not in
the data); claims of completeness (a quality-filtered ~50K subset of ~1.5M rows);
representative statistics about cinema (popularity-, English- and recency-skewed).

## Distribution and attribution

TMDB data and images come with attribution requirements. This product uses the TMDB
API but is not endorsed or certified by TMDB. (The catalog comes from a CSV export and
the poster URLs point at TMDB's CDN; the notice is used as TMDB prescribes it.) It is
repeated in the root `README.md`.

Posters are not stored or redistributed by this project. The backend builds the URL
`{catalog.poster_base_url}{catalog.poster_size}{poster_path}`
(`https://image.tmdb.org/t/p/` + `w500` + path, from `params.yaml`) and the browser
loads the image from TMDB's image CDN.

The licence of the CSV file itself is unrecorded (see Collection and source), so this
repository does not redistribute it; it is excluded by `.gitignore` and each
installation supplies its own copy. Before sharing the file or a derived database,
check the terms of the export and of TMDB's API terms of use.

## Maintenance

- Owner: the repository maintainer. There is no update schedule; the catalog is a
  snapshot of the CSV and does not follow TMDB.
- Coverage past 2023-09-30 needs a fresh CSV with a current `vote_count` snapshot; an old
  one cannot be fixed by lowering `catalog.min_vote_count`.
- To refresh: replace the CSV, run `import_catalog` (`--skip-existing` adds new
  titles but does not update existing rows) and `generate_embeddings`. Changing
  the import filter changes the catalog size and the popularity distribution, so
  re-check the evaluation afterwards.
- Changing the embedded text or the embedder triggers the provenance check at
  start-up and needs `generate_embeddings --all` (see
  [ML.md](ML.md#embedding-model)).

## The golden set

`data/golden_set.json` is the evaluation set, not part of the catalog, and it is
**draft**: 31 queries (21 dev, 10 test; 6 multi-turn, 6 with negated genres) with 193
relevant movies in total, written from general film knowledge **without access to the
catalog** and **not reviewed by a human**. Each is a `{title, year}` reference
resolved against the imported catalog at run time; references that do not resolve
(wrong year, different TMDB title, not in the filtered catalog) are reported and drop
out of that query's metrics. The lists are samples of good answers, not exhaustive, so
precision is a lower bound. The first 18 queries were migrated from an older test file
and the rest were added for the new evaluation. Review the labels before trusting any
number from it ([ADR 0007](adr/0007-evaluation-methodology.md), `TODOS.md`
"Evaluation").

## The label-query set (Reddit-Movie)

`data/label_queries.json` holds movie requests to be graded by hand, so the
evaluation does not rest on the draft golden set alone. It is built from
**Reddit-Movie** (`ZhankuiHe/reddit_movie_large_v1`, He et al. 2023,
[arXiv 2308.10053](https://arxiv.org/abs/2308.10053)) by
`manage.py import_label_queries` (`core/label_queries.py`). The command reads a
file the user downloaded; it never downloads the dataset, and the dataset itself
is not committed.

- **Format** (from the dataset card): `id2name.json` plus `{train, valid, test}.csv`,
  one row per conversation turn with the columns `conv_id`, `turn_id`, `turn_order`,
  `user_id`, `is_seeker`, `utc_time`, `upvotes`, `processed`, `raw`,
  `context_processed`, `context_raw`, `context_turn_ids`. `conv_id` is
  `<reddit thread id>_<path>/<paths>` (one conversation per path through the reply
  tree); `turn_order` 0 is the original post. `processed` has movie mentions replaced
  by IMDb ids (`tt0053779`), `raw` keeps the titles. The importer reads `conv_id`,
  `turn_id`, `turn_order`, `is_seeker`, `processed` and `raw`, and groups rows by
  thread, so the many paths of one thread yield one query.
- **Verified local format:** the downloaded test CSV contains 155,183 turns in
  5,952 threads. `processed`/`raw` cells are stringified `[role, text]` lists;
  the importer takes the last element and also accepts a plain `ROLE: text`.
  Missing headers/cells, invalid UTF-8 and malformed CSV fail with a file error.
  IMDb ids use `tt` followed by 7 or more digits.
- **Query and recs:** the query is the seeker's original post (`raw`, titles as
  written). `community_recs` are the TMDB ids of films the replies recommended,
  linked through the `imdb_id` column of the catalog CSV and limited to films the
  import keeps (`catalog.min_vote_count` etc.). They are a starting list for the
  grader, not labels: an upvoted reply can be wrong, and replies that name films
  without a catalog match are lost. All non-seeker turns, including nested replies,
  are discovery inputs; upvotes do not filter or rank them. The local source has
  19,439 non-seeker turns at depth >1 and 91,746 nonempty upvote cells, so these
  are real input features, deliberately not inferred relevance labels. IMDb IDs
  mentioned by the seeker anywhere in that thread are excluded even when echoed
  by replies; unrelated recommendations in the same reply remain.
- **Selection** (`evaluation.label_queries` in `params.yaml`): seeker post of
  30-600 characters, at least 3 distinct linked films, a seeded hash-ranked draw of
  120 threads, 33% to the initial test split. Existing split identities are retained
  by ID (existing output, then project `data/label_queries.json`, or `--existing`).
  Count/share changes therefore need not produce the requested share after retention.
  Re-import may still change selected membership; inspect the candidate by ID.
- **Review export:** the default is project `eval_runs/label_queries.candidate.json`,
  writable both on the host and at `/app/eval_runs/` in the container; `/app/data/`
  is read-only. Paths supplied explicitly are relative to the command working
  directory. From the repository root on the host:
  `cd backend && uv run python manage.py import_label_queries --source "../Reddit Movie Large Test.csv" --catalog "../TMDB Movie Dataset v11.csv"`.
  Container usage requires separately mounting the downloaded source/catalog,
  then supplying their actual mounted absolute paths with `--source` and `--catalog`;
  use `--out /app/eval_runs/label_queries.candidate.json`. Agents do not launch it.
  Never replace the reviewed query file until its exact candidate diff is approved.
- **2026-11-02 local audit:** all 115 retained queries match source text exactly,
  IDs are unique, all existing community IDs pass catalog membership, and splits
  remain 77 dev / 38 test. A separate candidate and per-ID audit live at
  `eval_runs/query-review-2026-11-02/`; reversible encoding/entity repairs are
  separated from community-list corrections. Seven ambiguous queries are retained
  with review notes and no semantic rewrite or deletion.
- **Known issues:** the card calls the text noisy and warns of toxic content; the
  data is Reddit English but not filtered for language or for being a request, so the
  user removes non-requests and non-English posts by hand before committing the file.
  Reddit users skew towards popular, English-language, recent films, like the catalog.
- **License:** the dataset card states no license. It says the data is "only for
  research use" (conversational recommendation, long-query retrieval). Building an
  offline evaluation set for a research project falls under that. The raw Reddit
  text is user content collected by Pushshift, so the file of selected posts is a
  small research excerpt, not a redistribution of the dataset; check Reddit's and
  the dataset's terms before publishing the repository with it. If that use is
  ever ruled out, the fallback is ReDial's first seeker utterances with
  `community_recs` from the movies the recommender mentioned, through the same
  `select_queries` interface.
