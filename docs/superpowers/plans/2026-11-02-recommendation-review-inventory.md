# Recommendation review finding inventory

Audit baseline: `5a8715d` (2026-11-02). This is an inventory, not implementation or quality acceptance. The tables preserve baseline evidence; the reconciliation below records final dispositions after Tasks 2–5. Static source/test inspection was used; no runtime tests, providers, network, model loads, startup or `.env` access occurred.

Source shorthand: **L** = `.superpowers/sdd/2026-10-25-recommendation-quality/progress.md`; **B/BR** = `broad-review.md` / `broad-rereview.md` in that directory; **Q** = `query-review.md`; **R1–R7** = corresponding `task-N-report.md`; **V5/V6/V7/VB** = `task-5-review.md`, `task-6-review.md`, `task-7-review.md`, `task-bootstrap-review.md`; **VR6** = `task-6-rereview.md`. All accessible task reports/reviews/rereviews, the broad fix report, ledger, Q, original recommendation-quality plan and current fixes plan were examined. Historical reports describe superseded intermediate states; current code/tests take precedence.

## Six historical branch-review Important findings

The six summaries survive in L; the original branch review itself is unavailable. These are independently classified, not merged into the missing fourteen minors.

| ID / original source | Affected behavior | Current evidence | Disposition | Owner |
|---|---|---|---|---|
| I-1 / L branch review | Forcing quality retrieval at configured fusion weight zero has no useful fusion mass | `candidate_generation.generate_candidates(rrf_weights=...)` and `graded_pool.capture_query` supply explicit positive variant weights; `test_candidate_generation.py::test_override_activates_quality_and_preserves_defaults` checks inclusion and unchanged defaults. Boolean force-on alone still uses configured zero, as designed; evaluation uses the override. | already fixed | T4 verify preservation |
| I-2 / L branch review | Missing quality membership blocks diagnostic export | Returned candidate dictionaries contain `channel_ranks` and `in_quality_channel`; `test_obscure_overrides_forced_quality_and_diagnostics` checks diagnostics. | already fixed | T4 verify preservation |
| I-3 / L branch review, R4 | Import defaults resolve relative to backend cwd, container data mount is read-only, filesystem errors escape | `import_label_queries.Command` defaults `--out` to `data/label_queries.json`, resolves relative paths directly, and performs mkdir/open without OSError conversion; R4 explicitly requires manual path workaround. | confirmed | T3 |
| I-4 / L branch review and deferred R4 | Seeker reference films echoed in replies inflate community recommendations | `label_queries._group_threads` skips seeker rows without retaining their IMDb IDs; reply matches are all accepted. | confirmed | T3 |
| I-5 / L branch review | Re-import/count changes shift existing dev/test identities | `select_queries` assigns test by index below `round(len(chosen)*test_share)`; existing identities are not inputs. Stable hashing alone cannot stabilize that threshold. | confirmed | T3 |
| I-6 / L branch review | Obscure quality removal raises absolute popularity weight | `scoring.effective_weights` proportionally rescales all remaining signals. `TestEffectiveWeights.test_effective_weights_drop_quality_for_obscure` explicitly checks sum 1 and retained ratios. Original plan requires renormalization and “turn quality off, nothing else”; these cannot both preserve every absolute weight. No evidence justifies changed weights. | not a defect | T4 document tradeoff and combined-filter coverage |

## Missing historical report

| Source | Affected behavior / evidence | Disposition | Owner |
|---|---|---|---|
| L: branch review records “14 minors (M-1..M-14)” | Actual text is absent. Searched available `.superpowers`/docs review artifacts and reachable `git log --all` report/review paths and review-subject history. Diff artifacts contain implementation changes, not the missing prose report. Do not equate the separately logged minors below with these unidentified fourteen; do not invent individual findings or mark them resolved. | source unavailable | T5 retain aggregate count 14 pending recovery |

## Separately logged minors and deferred findings

| ID / original source | Affected behavior | Current evidence | Disposition | Owner |
|---|---|---|---|---|
| M01 / L T1 | Positive votes with zero/default rating are nonneutral | `scoring._quality_score` passes rating 0 to `weighted_rating`, pulling below catalog mean; `test_missing_vote_fields_score_as_catalog_mean` does not cover positive-vote zero rating. | confirmed | T4 |
| M02 / L T1 | Evaluation log omits quality weight | `evaluate_scoring.py` “Weights:” line logs semantic, metadata, popularity only. | confirmed | T4 |
| M03 / L T1 | Repeated catalog-mean patches/import hygiene | `test_scoring.py` contains module autouse mean patch plus repeated per-test patches. | confirmed | T5 |
| M04 / L T1 | Missing quality settings raise bare KeyError | `settings._validate_quality` indexes `prior_votes`; startup indexes `quality`; `test_settings_validation.TestValidateQuality` covers values, not missing sections/keys. | confirmed | T4 |
| M05 / L T1 | API/core/ML describe three scoring signals | `docs/core.md:246`, `docs/ML.md:199` and API scoring descriptions remain stale despite later workflow documentation. | confirmed | T5 |
| M06 / L T2 | Semantic ordering duplicated | `generate_candidates` and `_quality_ranked_ids` each build cosine annotate/order/slice chains. | confirmed | T4 |
| M07 / L T2 | Mocked retrieval helper hygiene | `test_candidate_generation._mocked_retrieval` uses patch collections and function-local ExitStack imports; keep behavioral assertions during simplification. | confirmed | T5 |
| M08 / L T2 | Empty hard-filtered quality subset lacks PostgreSQL proof | Existing `TestQualityChannel` empty-subset tests mock retrieval; no permitted running DB is available. | user-run verification | T4/T5 |
| M09 / L T2 | Channel validation error coverage absent/incomplete | `_validate_quality_channel` reads nested mappings directly; existing settings tests do not exercise its missing/malformed channel sections. | confirmed | T4 |
| M10 / L T2 | Stale RRF two-channel wording | `params.yaml:207` says semantic and lexical; API retrieval description needs reconciliation with optional quality channel. | confirmed | T5 |
| M11 / L T2 | Long validation raise line | `_validate_quality_channel` quality-weight ValueError is still one overlong line (historical line number moved). | confirmed | T4 |
| M12 / L T3 | Direct offer/default-false and sync intent coverage missing | `movies.views._adds_to_offer` ignores False; sync `llm_client._extract_intent` carries wants_obscure. Existing broad chat/schema tests provide indirect protection, not direct assertions of these behaviors. | confirmed | T5 |
| M13 / L T3, R3 | Prompt-token estimate unmeasured | R3 measures +493 chars and estimates ~123 tokens; no real tokenizer/provider prompt-token evidence. Context remains 6144. | user-run verification | T5 user checklist |
| M14 / L T3 | All classify examples have obscure=false; recall unknown | `llm_client.py` few-shot examples preserve false-only pattern; mocked flag tests prove propagation, not recall. | user-run verification | T5 user checklist |
| M15 / L T3 | Long-lived config references task number | `params.yaml:66` says “wants_obscure (Task 3)”. | confirmed | T5 |
| M16 / L T4, R4 | Nested replies counted and upvotes ignored | `_group_threads` consumes every non-seeker turn without upvote filtering. Community lists are discovery aids, not relevance labels; policy needs explicit evidence/decision, not guessed labels. | confirmed | T3 document input policy |
| M17 / L T4 | Global CSV field-size setting duplicated | Both `label_queries.py` and importer set `csv.field_size_limit` on import. | confirmed | T3 |
| M18 / L T4 | Management command lacks integration test | Existing `test_label_queries.py` tests pure helpers; no import-label command call_command coverage. | confirmed | T3 |
| M19 / L T4 | Default output path backend-relative | Same behavior as I-3; retained explicitly to account for separately logged minor. | confirmed | T3 |
| M20 / L T4 | Split-share assertion tautological | `test_label_queries.py::test_split_share_within_one_query` uses small fixture and actual selected length; strengthen cardinality and split assertions. | confirmed | T3 |
| M21 / V5 and L T5 | Quadratic candidate channel-rank lookup | `candidate_generation.generate_candidates` performs per-candidate list membership/index across channels. | confirmed | T4 |
| M22 / VB and L bootstrap | Skipped/zero-eligible bootstrap branches not isolated | `test_paired_test_bootstrap_withholds_intervals_for_incomplete_comparison` sets incomplete coverage in both cases; branch implementation exists but separate coverage missing. | confirmed | T5 |

## Later reviews and earlier addressed findings

| ID / original source | Affected behavior | Current evidence | Disposition | Owner |
|---|---|---|---|---|
| B01 / V5 | Replay differs from disabled/single-item reranker production bypass and gate | `graded_pool.replay` has explicit bypass paths; `test_disabled_reranker_replay_matches_production`, `test_single_item_head_replay_matches_production` protect both blend variants. | already fixed | T5 preserve |
| B02 / V6, VR6 | Separate test run cannot import frozen dev choice | `graded_evaluation._load_frozen_proposal` validates recorded source and compatibility; `test_independent_dev_test_proposal_and_freeze`, `test_copied_proposal_preserves_source_dev_binding`. | already fixed | T5 preserve |
| B03 / V6, VR6 | Test exposure permits later refitting | `offline_report` validates proposal and writes source/target access locks before reading grades; `test_test_requires_choice_before_reading_grades`, `test_first_test_access_locks_even_incomplete_or_invalid_grades`, combined-grade and switch-proposal tests. | already fixed | T5 preserve |
| B04 / V6 | CV interval misread as final calibrated held-out uncertainty | Current Task 7 report and ML documentation distinguish CV baseline-gate ranking selection; held-out paired intervals implemented and tested in `test_paired_test_bootstrap_pairs_query_ids_and_reports_metric_samples`. | already fixed | T5 preserve |
| B05 / V7 | ADR index row outside Markdown table | `docs/adr/README.md` now has contiguous 0012 table row, as VB independently notes. | already fixed | T5 preserve |
| B06 / B, BR | Valid default-shaped intent rejected forever on resume | Export relies on recorded failure diagnostics rather than shape rejection; `test_graded_pool.py::test_default_shaped_success_and_recorded_failure`. | already fixed | T5 preserve |
| B07 / B, BR | Split drift accepted by frozen separate test runs | Frozen source dev/target test ID intersection rejected before grade access; `test_graded_evaluation.py::test_split_drift_rejected_before_grades`. Distinct-ID manual reidentification remains outside this guard. | already fixed | T5 preserve |
| B08 / B, BR | Adjacent missing-label worksheet cannot drive iterative report | `graded_evaluation.offline_report` validates/merges adjacent worksheet, atomically retains progress; `test_missing_worksheet_rerun_imports_and_preserves_grades` plus conflict/identity cases. | already fixed | T5 preserve |
| B09 / BR and L final | Worksheet-only grades lost in cross-mode/pilot reuse | `graded_pool.read_grade_source` reads original CSV only; `offline_report` imports worksheet separately. | confirmed | T2 |

## Query-data observations

Read-only recount of current untracked `data/label_queries.json`: **115 rows, 77 dev / 38 test**, not the historical Q audit's initial 120. Python JSON parsing and regex scans used only local files. Historical 37 mojibake IDs reduce to **36 present rows** because removed `t3_zwrf0g` was among them. The five HTML-entity rows remain. These are input repairs/review items, not grounds for automatic query removal.

| Original source | Affected behavior / current evidence | Disposition | Owner |
|---|---|---|---|
| Q: five unusable requests | IDs `t3_xetvfe`, `t3_z7z9ae`, `t3_zhiqm9`, `t3_zwrf0g`, `t3_zzln7h` are absent; 115-row count matches post-removal ledger. | already fixed | T3 preserve identities |
| Q: 37 mojibake observations | Current regex `â|Ã|ð|Â` matches 36 rows; compare source encoding before reversible repair. | confirmed | T3 |
| Q: five entity observations | Entity regex matches `t3_ylf1ns`, `t3_ysjtvi`, `t3_z0nm8e`, `t3_zvbzxn`, `t3_zw9n92`; last two include encoded zero-width layout artifacts. | confirmed | T3 |
| Q: t3_y5m7tk | Removed body; usable year-only title (1982). Review scope before deletion. | user-run verification | T3 query review |
| Q: t3_zle4fk | Removed body; ambient-lighting visual-style title remains. | user-run verification | T3 query review |
| Q: t3_z8d8qg | Christmas discussion/top-four list can express discovery intent. | user-run verification | T3 query review |
| Q: t3_zquy0n | Adaptation comparison with “Any others?”; literary-quality criterion exceeds catalog evidence. | user-run verification | T3 query review |
| Q: t3_ysiewd | Movie request also asks for similar director signature. | user-run verification | T3 query review |
| Q: t3_zzgiu5 | Scene discovery rather than whole-film request. | user-run verification | T3 query review |
| Q: t3_ylf1ns | Broad media-server request, little taste information. | user-run verification | T3 query review |
| Q/R4 | Catalog membership and exact source text/full-file format were not historically verified | Downloaded source CSV is present; source/catalog mapping audit and candidate diff belong to T3; do not mutate user original. Dataset card had no explicit license and said research use; no new license inference. | confirmed | T3 |

Mojibake IDs (current): `t3_xb9zan`, `t3_xc1ans`, `t3_xdk0tc`, `t3_xeitl5`, `t3_xgu3rr`, `t3_xru9jj`, `t3_xt9vlv`, `t3_xzmbyn`, `t3_y2bg3l`, `t3_y64fh6`, `t3_y6expz`, `t3_yaof94`, `t3_yaussb`, `t3_yb487s`, `t3_ydi8ca`, `t3_yoalbu`, `t3_yq2jwr`, `t3_ysjtvi`, `t3_ysjwsj`, `t3_ysnkft`, `t3_yvk6se`, `t3_yxrt1s`, `t3_yxsy3n`, `t3_z1p2xb`, `t3_z3irh5`, `t3_z3osn2`, `t3_z57xvb`, `t3_z785w9`, `t3_z93a8k`, `t3_zocdq8`, `t3_zquy0n`, `t3_ztu22i`, `t3_zvbzxn`, `t3_zy4sa2`, `t3_zzgiu5`, `t3_zzkolj`.

## Evidence-dependent concerns retained

| Original source | Affected behavior / evidence | Disposition | Owner |
|---|---|---|---|
| L/R1–R7/B/BR/VR6 | Catalog mean and hard-filtered quality retrieval on actual PostgreSQL; historical host runs had 78 DB-fixture errors/exclusions | Mocked tests establish local logic only; agents cannot start services or integration checks. | user-run verification | T5 checklist |
| R5/R7/B/BR | Provider batching equivalence, catalog fingerprint duration, pilot pool breadth/label burden, failures, GPU fit, VRAM/offload, context 6144, long-dialogue TTFT/latency, JSON/explanations | No real acceptance evidence; preserve pinned models/dimensions and baseline. | user-run verification | T5 checklist |
| R6/R7/V7/VB and original plan Tasks 7–8 | Human grades, dev selection, independent BGE/Voyage gate calibration, frozen held-out nDCG/junk/empty/coverage/bootstrap acceptance; cleanup requires both modes accepted | Voyage null gate remains uncalibrated; saved-pool metrics and LLM judge are not catalog-wide quality truth. Task 8 remains deferred. | user-run verification | T5 record pending |
| VR6 | Source-dev path retention, conservative attempted-access locking; artifact tamper/concurrent writers unsupported | Current workflow documents source retention and sequential local artifact assumptions; no requirement for adversarial tamper protection. | not a defect | T5 preserve limits |

Preservation: starting tree contained only user-modified original plan and untracked source CSV/query JSON. This task creates this inventory and its report only. No existing plans, query records, grades, manifests or source datasets were edited. `git diff --check` validates this documentation; later task edits are independently owned.


## Final reconciliation (2026-11-02)

The baseline descriptions above are historical evidence, not claims that fixed code remains defective. Reports under `.superpowers/sdd/2026-11-02-recommendation-review-fixes/` provide exact changes and validation.

| Findings | Final disposition / evidence |
|---|---|
| I-1, I-2 | Already fixed; Task 4 guarded retrieval coverage preserved override activation, obscure precedence and diagnostic membership. |
| I-3, I-4, I-5; M16–M20 | Fixed by Task 3: writable candidate default/actionable input errors, seeker-echo exclusion, retained splits, explicit nested/upvote policy, one CSV limit setter, command tests and exact cardinality/split assertions. 41 focused tests passed. |
| I-6 | Not a defect: proportional renormalization preserves ratios and necessarily changes other absolute weights; Task 4 combined-filter coverage passes. |
| M01, M02, M04, M06, M09, M11, M21 | Fixed by Task 4: neutral absent ratings, quality logging, actionable finite config validation, shared cosine order, rank maps. 360 targeted passes/15 DB exclusions; settings follow-up 193 passes. |
| M03, M07 | Fixed by Task 5: redundant mean patches removed, intentional unavailable-mean override retained; ExitStack imported once at module scope with existing mocked retrieval behavioral assertions preserved. |
| M05, M10, M15 | Fixed by Task 5: four-signal API/core/ML descriptions, optional third retrieval channel and obscure-weight tradeoff documented; stale task reference removed. Historical three-signal tuning evidence remains explicitly historical. |
| M12 | Fixed by Task 5: direct `_adds_to_offer` cases for false/default and explicit true, and synchronous `_extract_intent` true/false/absent cases. |
| M20, M22 | Exact split counts/cardinality covered by Task 3; Task 5 independently isolates incomplete coverage, skipped query and zero-eligible bootstrap withholding. |
| B01–B08 | Existing regression behavior preserved by complete permitted suite, including replay bypass/gate, dev/test binding and access locks, split drift and iterative worksheets. |
| B09 | Fixed by Task 2: original-path reuse merges/validates adjacent worksheet with exact snapshot identity and file/line diagnostics; 170 focused tests passed. |
| M08, M13, M14 and evidence-dependent concerns | Pending user-run PostgreSQL, real token/recall, provider batch, GPU/context/latency and human graded quality evidence. Exact procedures in `docs/infrastructure.md` user-only review checks. No mocked test is quality acceptance. |
| Query encoding/entities/source membership | Task 3 prepared fingerprinted per-ID audit and candidate: 115 identities/splits retained, 36 encoding/five entity repairs, 32 community changes; source/catalog consistency established locally. Original JSON/CSVs untouched; candidate adoption and seven semantic-scope decisions await user review. |
| Fourteen unidentified historical minors | Source unavailable. No individual finding invented or closed; original prose recovery remains outstanding. |
| Original recommendation-quality Task 8 | Deferred until both modes have accepted human-quality/operational evidence; Voyage null gate remains uncalibrated. |

Task 5 fresh verification: complete backend suite with socket connect/connect_ex/create_connection/getaddrinfo blocked before pytest: **1406 passed, 78 `django_db` tests deselected**. Targeted hygiene modules: **356 passed, 33 deselected**. Both effective profiles and weight sums are exercised by `test_effective_scoring_profiles_validate_and_cloud_override_is_isolated` in the full run. Real DB/API/GPU checks remain pending. No original plans, source datasets, labels, manifests or snapshots were changed. Tasks 1–4 were committed by the controller after worker sandbox restrictions; Task 5 is handed back uncommitted. No pushes.
