"""Run offline evaluation of the recommendation pipeline against the golden set.

Usage:
  python manage.py evaluate_scoring                        # e2e intent, test split
  python manage.py evaluate_scoring --intent oracle        # labelled intent, no LLM parse
  python manage.py evaluate_scoring --split dev            # dev | test | all
  python manage.py evaluate_scoring --llm-judge            # + LLM judge (secondary signal)
  python manage.py evaluate_scoring --sweep                # weight grid search, dev split only
  python manage.py evaluate_scoring --weights 0.5,0.3,0.2  # custom weights
  python manage.py evaluate_scoring --gate-report          # calibrate reranking.min_score, dev split only
  python manage.py evaluate_scoring --output run.json      # run record path

What makes this an honest measurement:
  - Relevance is membership in the golden set's labelled movies (with grades),
    never genre overlap -- the `metadata` signal computes that overlap from the
    same genres, so it would grade the ranker against its own input.
  - `--intent e2e` (default) runs the production chat path: the combined
    classify+parse LLM call, embedding its `semantic_query`. `oracle` swaps
    in the labelled intent to isolate retrieval from parsing.
  - Retrieval is `core.pipeline.generate_and_score`, the function ChatView
    runs, so eval and production cannot drift.
  - Multi-turn entries get a session vector built the way production builds it.
  - `--sweep` only ever selects on the dev split; the plain eval defaults to
    the held-out test split.
  - `--gate-report` runs with the relevance gate off and prints how rerank_score
    separates labelled hits from the rest, plus what each candidate threshold
    would drop. It informs a human choice of reranking.min_score and never
    writes params.yaml.
  - Every run writes a JSON record (git sha, params.yaml hash, options,
    per-query results, counters) so a number can be traced to what produced it.
"""

import asyncio
import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from core.embedding_service import encode_query
from core.evaluation import (
    aggregate_metrics,
    compute_genre_frequency,
    coverage,
    evaluate_query,
    gate_report,
    llm_relevance_score,
)
from core.golden_set import (
    default_golden_set_path,
    load_golden_set,
    reference_label,
    resolve_references,
    select_split,
)
from core.ollama_client import MessageIntent, aclassify_and_parse, ais_available
from core.pipeline import generate_and_score
from core.session_manager import update_preference_vector, vector_for_scoring
from movies.models import Movie

logger = logging.getLogger(__name__)

ACCURACY_KEYS = ("precision_at_k", "recall_at_k", "ndcg_at_k")
# Categories the chat view answers without searching. In e2e mode a golden
# query classified as one of these is still searched (so it still gets
# metrics) but counted as misclassified -- in production it would have
# returned no recommendations at all.
NON_SEARCH_CATEGORIES = ("follow_up", "general_chat")
OWN_OPTIONS = (
    "golden_set", "split", "intent", "k", "sweep", "gate_report", "weights", "llm_judge", "output",
)


def _catalog_stats() -> tuple[dict[str, float], int]:
    """Genre frequencies (for novelty) and catalog size (for coverage)."""
    catalog_genres = list(Movie.objects.values_list("genres", flat=True))
    return compute_genre_frequency(catalog_genres), len(catalog_genres)


def _scoring_weights(semantic: float, metadata: float, session: float) -> dict:
    """Full scoring weights from the three signals this command varies.

    The three are taken as relative proportions and scaled to `1 - popularity`,
    with popularity held at its configured value, so the result sums to 1.0 and
    can be pasted into params.yaml as-is (settings refuses weights that do not
    sum to 1). Session takes the rounding remainder to keep the sum exact.
    """
    popularity = settings.SCORE_WEIGHTS.get("popularity", 0.0)
    total = semantic + metadata + session
    if total <= 0:
        raise CommandError("weights must have a positive sum")
    scale = (1.0 - popularity) / total
    sem, met = round(semantic * scale, 4), round(metadata * scale, 4)
    return {
        "semantic": sem, "metadata": met,
        "session": round(1.0 - popularity - sem - met, 4),
        "popularity": popularity,
    }


def _git_state() -> dict:
    """Commit sha and dirty flag of the code that ran, or nulls when git or the
    repository is unavailable (e.g. inside the Docker image, which has no .git)."""
    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=settings.BASE_DIR, capture_output=True, text=True,
            check=True, timeout=10,
        ).stdout

    try:
        commit = git("rev-parse", "HEAD").strip()
        dirty = bool(git("status", "--porcelain").strip())
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit or None, "dirty": dirty}


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_default(value):
    # numpy scalars from metric code
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


@dataclass
class PreparedQuery:
    """Everything about a golden entry that does not depend on the scoring
    weights -- computed once, so a sweep does not re-run the LLM per config."""

    entry: dict
    intent: dict
    query_embedding: list[float]
    session_vector: list[float] | None
    category: str
    predicted_category: str | None
    # e2e only (None in oracle): the parsed intent has exactly the shape
    # aclassify_and_parse returns when it gives up -- see _is_fallback_shape.
    possible_parse_fallback: bool | None
    grades: dict[int, int]
    unresolved: list[dict]

    @property
    def misclassified(self) -> bool:
        return self.predicted_category in NON_SEARCH_CATEGORIES


class Command(BaseCommand):
    help = "Evaluate recommendation quality on the golden set"

    def add_arguments(self, parser):
        parser.add_argument(
            "--golden-set",
            type=str,
            default=None,
            help="Path to the golden set JSON (default: data/golden_set.json)",
        )
        parser.add_argument(
            "--split",
            choices=["dev", "test", "all"],
            default=None,
            help="Golden-set split to run (default: test; --sweep always uses dev)",
        )
        parser.add_argument(
            "--intent",
            choices=["e2e", "oracle"],
            default="e2e",
            help="e2e: parse intent with the production LLM call (default); "
                 "oracle: use each entry's labelled oracle_intent",
        )
        parser.add_argument(
            "--k",
            type=int,
            default=5,
            help="Number of recommendations to evaluate (default: 5)",
        )
        parser.add_argument(
            "--sweep",
            action="store_true",
            help="Run grid search over weight space (dev split only)",
        )
        parser.add_argument(
            "--gate-report",
            action="store_true",
            help="Calibration report for reranking.min_score: with the gate off, the "
                 "rerank_score of hits vs non-hits and the effect of candidate thresholds "
                 "(dev split only)",
        )
        parser.add_argument(
            "--weights",
            type=str,
            default=None,
            help=(
                "Custom weights as comma-separated values: semantic,metadata,session. "
                "Taken as proportions and scaled to 1 - popularity (popularity stays at "
                "its params.yaml value), so the result sums to 1."
            ),
        )
        parser.add_argument(
            "--llm-judge",
            action="store_true",
            help="Also score results with the Ollama LLM judge (slow: ~2s/movie). "
                 "Same model family as the system, so a secondary signal only",
        )
        parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="Run record path (default: eval_runs/<mode>-<timestamp>.json "
                 "under the working directory)",
        )

    def handle(self, *args, **options):
        k = options["k"]
        if k < 1:
            raise CommandError("--k must be at least 1")
        sweep = options["sweep"]
        report_gate = options["gate_report"]
        if sweep and report_gate:
            raise CommandError("--sweep and --gate-report are separate runs; pick one.")
        split = options["split"] or ("dev" if sweep or report_gate else "test")
        if sweep and split != "dev":
            raise CommandError(
                f"--sweep selects weights, so it only runs on the dev split (got --split {split}); "
                "the test split is held out for the final report."
            )
        if report_gate and split != "dev":
            raise CommandError(
                f"--gate-report picks a threshold, so it only runs on the dev split (got --split {split}); "
                "the test split is held out for the final report."
            )
        intent_mode = options["intent"]
        if intent_mode == "e2e" and not asyncio.run(ais_available()):
            # aclassify_and_parse never raises -- on a failed request it returns
            # the raw message with no filters -- so without this check an
            # unreachable Ollama would produce a plausible-looking "e2e" run
            # that never parsed anything.
            raise CommandError(
                f"--intent e2e needs Ollama, which is not reachable at {settings.OLLAMA_BASE_URL}. "
                "Start it, or run --intent oracle to evaluate retrieval alone."
            )
        use_llm = options["llm_judge"]
        weights = self._parse_weights(options["weights"])

        golden_path = Path(options["golden_set"]) if options["golden_set"] else default_golden_set_path()
        entries = select_split(load_golden_set(golden_path), split)
        if not entries:
            raise CommandError(f"no golden-set entries in split {split!r}")
        self.stdout.write(f"Loaded {len(entries)} {split} queries from {golden_path} (intent: {intent_mode})")
        if use_llm:
            self.stdout.write("LLM-as-judge enabled (this will be slow)")

        self.genre_freq, self.catalog_size = _catalog_stats()
        started = datetime.now(timezone.utc)

        prepared = [self._prepare(entry, intent_mode) for entry in entries]
        self._report_unresolved(prepared)

        record = {
            "timestamp": started.isoformat(),
            "mode": "sweep" if sweep else "gate_report" if report_gate else "eval",
            "git": _git_state(),
            "params_file": str(settings.PARAMS_FILE),
            "params_sha256": _sha256(settings.PARAMS_FILE),
            "golden_set": str(golden_path),
            "golden_set_sha256": _sha256(golden_path),
            "options": {key: options[key] for key in OWN_OPTIONS},
            "intent_mode": intent_mode,
            "split": split,
            "k": k,
            "llm_judge": use_llm,
            "catalog_size": self.catalog_size,
        }
        if sweep:
            record["configs"] = self._run_sweep(prepared, k, use_llm)
            record["best_weights"] = record["configs"][0]["weights"]
        else:
            record["weights"] = weights
            # The report needs every returned movie's score, including the ones
            # the gate would drop, so its run is ungated.
            record.update(self._run_eval(prepared, k, weights, use_llm, apply_gate=not report_gate))
            if report_gate:
                record["gate_report"] = gate_report(record["queries"])
                self._write_gate_report(record["gate_report"])

        output = Path(options["output"]) if options["output"] else (
            Path("eval_runs") / f"{record['mode']}-{started.strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=_json_default))
        self.stdout.write(f"\nRun record written to {output}")

    def _parse_weights(self, weights_str):
        if not weights_str:
            return dict(settings.SCORE_WEIGHTS)
        try:
            parts = [float(x) for x in weights_str.split(",")]
        except ValueError:
            parts = []
        if len(parts) != 3:
            raise CommandError("--weights needs three numbers: semantic,metadata,session")
        # popularity is held at its configured value -- --weights only varies
        # the three signals it always has, so it doesn't silently zero popularity.
        # The three are scaled to 1 - popularity so the weights stay pasteable.
        return _scoring_weights(*parts)

    # --- Per-entry preparation (weight-independent) ---

    def _prepare(self, entry: dict, intent_mode: str) -> PreparedQuery:
        query = entry["query"]
        context = entry.get("context", [])
        if intent_mode == "e2e":
            # Mirrors ChatView: the combined classify+parse call with the prior
            # turns as history, then embed its rewritten semantic_query. The
            # golden set holds user messages only, so history is user turns.
            history = [{"role": "user", "content": message} for message in context]
            message_intent = asyncio.run(aclassify_and_parse(query, history))
            possible_fallback = self._is_fallback_shape(message_intent, query)
            predicted = message_intent.category
            category = predicted
            query_embedding = encode_query(message_intent.semantic_query)
            intent = message_intent.model_dump(exclude={"category", "semantic_query"})
        else:
            predicted = None
            possible_fallback = None
            category = entry.get("category", "new_search")
            query_embedding = encode_query(query)
            intent = dict(entry.get("oracle_intent", {}))

        grades, unresolved = resolve_references(entry["relevant"])
        return PreparedQuery(
            entry=entry,
            intent=intent,
            query_embedding=query_embedding,
            session_vector=self._session_vector(context, query_embedding, category),
            category=category,
            predicted_category=predicted,
            possible_parse_fallback=possible_fallback,
            grades=grades,
            unresolved=unresolved,
        )

    @staticmethod
    def _is_fallback_shape(intent: MessageIntent, query: str) -> bool:
        """True when the parse is indistinguishable from aclassify_and_parse's
        failure fallback: new_search, semantic_query == the raw message, no
        filters. A genuine parse can look like this too (a vague query the
        model left as-is), hence "possible" -- but a run where most entries
        match is a run where parsing did not happen."""
        return intent == MessageIntent(category="new_search", semantic_query=query)

    @staticmethod
    def _session_vector(context: list[str], query_embedding: list[float], category: str):
        """The preference vector production would score this turn with: each
        prior message folded in by EMA, then `vector_for_scoring` for the final
        message. Prior turns use the default alpha -- their categories are not
        labelled. No context means a first turn: None, as in production."""
        current = None
        for message in context:
            current = update_preference_vector(current, encode_query(message))
        return vector_for_scoring(current, query_embedding, category)

    def _report_unresolved(self, prepared: list[PreparedQuery]) -> None:
        total = sum(len(p.unresolved) for p in prepared)
        if not total:
            return
        self.stdout.write(self.style.WARNING(
            f"\n{total} golden reference(s) did not resolve to a catalog movie -- "
            "they are excluded from recall/nDCG for their query. Fix the label or check the catalog:"
        ))
        for p in prepared:
            if p.unresolved:
                labels = ", ".join(reference_label(ref) for ref in p.unresolved)
                note = "  [NO reference resolved: query not scored]" if not p.grades else ""
                self.stdout.write(self.style.WARNING(f"  {p.entry['id']}: {labels}{note}"))
        self.stdout.write("")

    # --- Scoring (per weight configuration) ---

    @staticmethod
    def _judge_query(entry: dict) -> str:
        # A refinement like "something funnier" means nothing on its own; the
        # judge sees the conversation the system saw.
        return " -> ".join([*entry.get("context", []), entry["query"]])

    def _score_query(
        self, p: PreparedQuery, k: int, weights: dict, use_llm: bool, apply_gate: bool = True,
    ) -> dict:
        entry = p.entry
        top, pool = generate_and_score(
            p.query_embedding, p.intent, p.session_vector,
            query_text=entry["query"], weights=weights, top_n=k, apply_gate=apply_gate,
        )
        recommended_ids = [m["tmdb_id"] for m in top]

        negated = set(entry.get("negated_genres", []))
        negation_violations = sum(1 for m in top if negated & set(m.get("genres", [])))

        judge_scores = None
        if use_llm:
            judge_query = self._judge_query(entry)
            judge_scores = [llm_relevance_score(judge_query, m) for m in top]

        metrics = evaluate_query(
            recommended_ids,
            p.grades,
            [m["embedding"] for m in top if m.get("embedding") is not None],
            recommended_genres=[m.get("genres", []) for m in top],
            genre_freq=self.genre_freq,
            judge_scores=judge_scores,
            k=k,
        )
        unscorable = not p.grades
        if unscorable:
            # No resolved label means no ground truth: a 0.0 here would blame the
            # ranker for a labelling/catalog gap. Reported and counted instead.
            for key in ACCURACY_KEYS:
                metrics.pop(key, None)
        metrics["negation_violations"] = negation_violations

        result = {
            "id": entry["id"],
            "query": entry["query"],
            "split": entry["split"],
            "context": entry.get("context", []),
            "category": p.category,
            "intent": p.intent,
            "recommended": [
                {
                    "tmdb_id": m["tmdb_id"], "title": m["serial_name"],
                    "release_date": m.get("release_date"),
                    # None when the reranker did not score this request.
                    "rerank_score": m.get("rerank_score"),
                    "hit": m["tmdb_id"] in p.grades,
                }
                for m in top
            ],
            # Best cross-encoder score in the whole reranked pool: with the
            # gate on, a query is empty exactly when this is below min_score.
            "pool_max_rerank_score": max(
                (c["rerank_score"] for c in pool if c.get("rerank_score") is not None), default=None,
            ),
            "hits": [tmdb_id for tmdb_id in recommended_ids if tmdb_id in p.grades],
            "resolved_refs": len(p.grades),
            "unresolved_refs": [reference_label(ref) for ref in p.unresolved],
            "unscorable": unscorable,
            "metrics": metrics,
        }
        if p.predicted_category is not None:
            result["predicted_category"] = p.predicted_category
            result["misclassified"] = p.misclassified
            result["possible_parse_fallback"] = p.possible_parse_fallback
        if judge_scores is not None:
            result["judge_scores"] = judge_scores
            result["judge_failures"] = sum(1 for s in judge_scores if s is None)
        return result

    def _evaluate(
        self, prepared: list[PreparedQuery], k: int, weights: dict, use_llm: bool,
        apply_gate: bool = True,
    ) -> dict:
        queries = [self._score_query(p, k, weights, use_llm, apply_gate) for p in prepared]
        aggregate = aggregate_metrics([q["metrics"] for q in queries])
        aggregate["coverage"] = coverage(
            [[r["tmdb_id"] for r in q["recommended"]] for q in queries], self.catalog_size,
        )
        counters = {
            "queries": len(queries),
            "scored_queries": sum(1 for q in queries if not q["unscorable"]),
            "unscorable_queries": sum(1 for q in queries if q["unscorable"]),
            "unresolved_refs": sum(len(p.unresolved) for p in prepared),
            "judge_failures": sum(q.get("judge_failures", 0) for q in queries),
            "empty_results": sum(1 for q in queries if not q["recommended"]),
            "misclassified": sum(1 for p in prepared if p.misclassified),
            "possible_parse_fallbacks": sum(1 for p in prepared if p.possible_parse_fallback),
        }
        return {"queries": queries, "aggregate": aggregate, "counters": counters}

    def _write_counters(self, counters: dict) -> None:
        self.stdout.write("COUNTERS:")
        for key, value in counters.items():
            self.stdout.write(f"  {key}: {value}")
        if counters["possible_parse_fallbacks"]:
            self.stdout.write(self.style.WARNING(
                f"{counters['possible_parse_fallbacks']}/{counters['queries']} e2e intents match the "
                "classify+parse failure fallback (raw query, no filters) -- check the Ollama logs "
                "before trusting this as an e2e result."
            ))

    def _run_eval(self, prepared, k, weights, use_llm, apply_gate=True) -> dict:
        self.stdout.write(
            f"Weights: semantic={weights['semantic']}, metadata={weights['metadata']}, "
            f"session={weights['session']}, popularity={weights['popularity']}"
        )
        result = self._evaluate(prepared, k, weights, use_llm, apply_gate)

        for q in result["queries"]:
            m = q["metrics"]
            if q["unscorable"]:
                accuracy = "UNSCORED (no resolved labels)"
            else:
                accuracy = (
                    f"P@{k}={m['precision_at_k']:.2f} R@{k}={m['recall_at_k']:.2f} "
                    f"nDCG={m['ndcg_at_k']:.2f}"
                )
            extras = ""
            if "llm_relevance" in m:
                extras += f" LLM={m['llm_relevance']:.1f}"
            if "judge_ndcg" in m:
                extras += f" jNDCG={m['judge_ndcg']:.2f}"
            if m["negation_violations"]:
                extras += f" neg={m['negation_violations']}"
            if q.get("misclassified"):
                extras += f" MISCLASSIFIED({q['predicted_category']})"
            if q.get("possible_parse_fallback"):
                extras += " POSSIBLE_PARSE_FALLBACK"
            if q["unresolved_refs"]:
                extras += f" unresolved={len(q['unresolved_refs'])}"
            self.stdout.write(f"  {q['id'][:32]:32s} {accuracy} Div={m.get('diversity', 0):.2f}{extras}")

        self.stdout.write("\n" + "=" * 70)
        self.stdout.write("AGGREGATE METRICS:")
        for key, value in result["aggregate"].items():
            self.stdout.write(f"  {key}: {value:.4f}")
        self._write_counters(result["counters"])
        return result

    def _write_gate_report(self, report: dict) -> None:
        self.stdout.write("\n" + "=" * 70)
        self.stdout.write(
            "GATE CALIBRATION (reranking.min_score; gate off for this run, dev split)"
        )
        considered = report["queries_considered"]
        self.stdout.write(
            f"  queries: {considered} with scores, {report['queries_unscorable']} unscorable "
            f"(no resolved labels), {report['queries_without_scores']} without rerank scores"
        )
        if not considered:
            self.stdout.write(self.style.WARNING(
                "No rerank scores were collected -- the reranker is disabled, failed to load, "
                "or failed to predict. Fix that before calibrating anything."
            ))
            return
        self.stdout.write(f"\n  {'rerank_score':14s} {'count':>6} {'min':>8} {'q1':>8} {'median':>8} {'q3':>8} {'max':>8}")
        for label, key in (("hits", "hits"), ("non-hits", "non_hits")):
            d = report[key]
            cells = "".join(f" {d[c]:>8.4f}" if d[c] is not None else f" {'-':>8}" for c in ("min", "q1", "median", "q3", "max"))
            self.stdout.write(f"  {label:14s} {d['count']:>6}{cells}")

        def pct(share):
            return f"{share:>7.1%}" if share is not None else f"{'-':>7}"

        self.stdout.write(
            f"\n  {'min_score':>9} {'hits dropped':>13} {'non-hits dropped':>17} {'queries emptied':>16}"
        )
        for row in report["thresholds"]:
            self.stdout.write(
                f"  {row['min_score']:>9.3f} {pct(row['hits_dropped_share']):>13} "
                f"{pct(row['non_hits_dropped_share']):>17} {pct(row['queries_emptied_share']):>16}"
            )
        self.stdout.write(
            "\nPick a value by hand (a threshold that drops few hits but many non-hits), set "
            "reranking.min_score in params.yaml, and judge it on the test split. This command "
            "never writes params.yaml."
        )

    def _run_sweep(self, prepared, k, use_llm) -> list[dict]:
        if use_llm:
            grid = [
                (0.7, 0.2, 0.1),
                (0.6, 0.2, 0.2),
                (0.5, 0.3, 0.2),
                (0.4, 0.3, 0.3),
                (0.3, 0.4, 0.3),
                (0.5, 0.2, 0.3),
            ]
            self.stdout.write(
                f"Running weight grid search with LLM judge ({len(grid)} configs, "
                f"~{len(grid) * len(prepared) * k * 2}s estimated)..."
            )
        else:
            grid = [
                (sem, met, round(1.0 - sem - met, 2))
                for sem in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
                for met in [0.1, 0.2, 0.3, 0.4]
                if round(1.0 - sem - met, 2) >= 0
            ]
            self.stdout.write(f"Running weight grid search ({len(grid)} configs)...")

        configs = []
        for i, (sem, met, ses) in enumerate(grid):
            weights = _scoring_weights(sem, met, ses)
            if use_llm:
                self.stdout.write(
                    f"  Config {i+1}/{len(grid)}: sem={weights['semantic']} "
                    f"met={weights['metadata']} ses={weights['session']}..."
                )
            configs.append({"weights": weights, **self._evaluate(prepared, k, weights, use_llm)})

        # Ranked by golden-set nDCG, never by the judge: the judge is the same
        # model family as the system, so selecting on it is circular.
        configs.sort(key=lambda c: c["aggregate"].get("ndcg_at_k", 0), reverse=True)

        header_llm = f" {'LLM':>6} {'jNDCG':>6}" if use_llm else ""
        self.stdout.write(f"\nWeight configurations by ndcg_at_k (dev split, k={k}):")
        self.stdout.write(
            f"{'Semantic':>8} {'Meta':>6} {'Session':>8} {'P@k':>6} {'R@k':>6} {'NDCG':>6} {'Nov':>6}{header_llm}"
        )
        self.stdout.write("-" * (55 + (14 if use_llm else 0)))
        for config in configs:
            w, m = config["weights"], config["aggregate"]
            llm_cols = (
                f" {m.get('llm_relevance', 0):>6.2f} {m.get('judge_ndcg', 0):>6.3f}" if use_llm else ""
            )
            self.stdout.write(
                f"{w['semantic']:>8.2f} {w['metadata']:>6.2f} {w['session']:>8.2f} "
                f"{m.get('precision_at_k', 0):>6.3f} {m.get('recall_at_k', 0):>6.3f} "
                f"{m.get('ndcg_at_k', 0):>6.3f} {m.get('novelty', 0):>6.3f}{llm_cols}"
            )
        self.stdout.write("")
        self._write_counters(configs[0]["counters"])
        return configs
