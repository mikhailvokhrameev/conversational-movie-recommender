# 0012: Calibrate quality separately by inference mode on saved human-graded pools

**Status:** Proposed — workflow implemented; settings and quality acceptance pending.

## Context

Local BGE and cloud Voyage produce different raw relevance scores. No new graded
GPU/API snapshot, fitted configuration or held-out result is available. The existing
local binary-label threshold is not cloud calibration. One assessor's grades describe
that assessor's judgments, not broad user satisfaction.

## Decision

Use mode-specific saved dev pools and human grades, reuse matching query/movie
grades across providers, and compare shared scoring weights before considering cloud
scoring overrides. All selected tunables remain in `params.yaml`; cloud overrides
belong under `runtime_profiles.openrouter`. Calibrate absolute gates separately;
Voyage's null threshold remains uncalibrated.

Freeze dev proposals before reading separate held-out test grades. Replay each
mode's proposal and frozen baseline on identical mode-specific test snapshots.
Require complete judged top-five comparisons, lower junk@5 and no reduction in
nDCG@5; report paired query-bootstrap intervals, empty rates, coverage and sample
size. The offline test report computes deterministic paired selected-minus-baseline
95% intervals for all six reported metrics, with per-metric eligible query counts;
incomplete or skipped-query comparisons withhold intervals. This compares the
final frozen proposal including its dev-selected gate; dev CV intervals cover
ranking selection at the baseline gate only. Failed criteria retain the baseline. Unexpected extra empty answers or provider
failures in user-run golden/interactive checks block that mode's acceptance.

The [workflow](../ML.md#graded-saved-pool-workflow-quality-calibration-pending)
documents the implemented commands and frozen proposal handoff. Existing evaluation
commands remain available. Offline tests establish mechanics, not quality.

## Consequences

No parameter change is justified yet. Record actual mode, catalog/index, models,
dev/test sizes, grades, CV/test results, gate and limitations when evidence exists.
Pool-based judgments cannot establish catalog-wide recall. Retain source dev
artifacts for test provenance validation. Real inference, GPU/VRAM, latency and
behaviour checks remain pending user evidence; test grades must never select settings.
