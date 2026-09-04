"""Run offline evaluation of the recommendation scoring pipeline.

Usage:
  python manage.py evaluate_scoring                        # genre-based eval
  python manage.py evaluate_scoring --llm-judge            # + LLM relevance scoring
  python manage.py evaluate_scoring --sweep                # grid search over weights
  python manage.py evaluate_scoring --weights 0.5,0.3,0.2  # custom weights
"""

import json
import logging

from django.conf import settings
from django.core.management.base import BaseCommand

from core.candidate_generation import generate_candidates
from core.embedding_service import encode_query
from core.evaluation import (
    aggregate_metrics,
    compute_genre_frequency,
    coverage,
    evaluate_query,
    llm_relevance_score,
)
from core.scoring import mmr_diversify, score_candidates
from movies.models import Movie

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Evaluate recommendation scoring quality on the test query set"

    def add_arguments(self, parser):
        parser.add_argument(
            "--test-set",
            type=str,
            default="data/test_queries.json",
            help="Path to test queries JSON file",
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
            help="Run grid search over weight space",
        )
        parser.add_argument(
            "--weights",
            type=str,
            default=None,
            help="Custom weights as comma-separated values: semantic,metadata,session",
        )
        parser.add_argument(
            "--llm-judge",
            action="store_true",
            help="Use Ollama LLM-as-judge for graded relevance scoring (slow: ~2s/movie)",
        )

    def handle(self, *args, **options):
        with open(options["test_set"]) as f:
            test_set = json.load(f)

        k = options["k"]
        use_llm = options["llm_judge"]
        self.stdout.write(f"Loaded {len(test_set)} test queries")
        if use_llm:
            self.stdout.write("LLM-as-judge enabled (this will be slow)")

        catalog_genres = list(
            Movie.objects.values_list("genres", flat=True)
        )
        self.genre_freq = compute_genre_frequency(catalog_genres)
        self.catalog_size = Movie.objects.count()

        self._use_llm = use_llm

        if options["sweep"]:
            self._run_sweep(test_set, k)
        else:
            weights = self._parse_weights(options["weights"])
            self._run_eval(test_set, k, weights, use_llm)

    def _parse_weights(self, weights_str):
        if weights_str:
            parts = [float(x) for x in weights_str.split(",")]
            return {"semantic": parts[0], "metadata": parts[1], "session": parts[2]}
        return settings.SCORE_WEIGHTS

    def _build_intent(self, test_case):
        intent = {"genres": list(test_case.get("relevant_genres", []))}
        if "negated_genres" in test_case:
            intent["negations"] = test_case["negated_genres"]
        if "relevant_content_type" in test_case:
            intent["content_type"] = test_case["relevant_content_type"]
        return intent

    def _score_query(self, test_case, k, weights, use_llm=False):
        query = test_case["query"]
        relevant_genres = set(test_case.get("relevant_genres", []))
        relevant_content_type = test_case.get("relevant_content_type")
        negated_genres = set(test_case.get("negated_genres", []))

        query_embedding = encode_query(query)
        intent = self._build_intent(test_case)
        candidates = generate_candidates(query_embedding, intent)

        scored = score_candidates(candidates, query_embedding, intent, weights=weights)
        scored.sort(key=lambda x: x["total"], reverse=True)
        diversified = mmr_diversify(scored, top_n=k)

        recommended_titles = [m["serial_name"] for m in diversified]
        recommended_embeddings = [
            m["embedding"] for m in diversified if m.get("embedding") is not None
        ]
        recommended_genres = [m.get("genres", []) for m in diversified]

        hits = []
        negation_violations = 0
        content_type_violations = 0
        for m in diversified:
            movie_genres = set(m.get("genres", []))
            is_relevant = False

            if relevant_genres and (relevant_genres & movie_genres):
                is_relevant = True
            if relevant_content_type and m.get("content_type") == relevant_content_type:
                is_relevant = True

            if negated_genres and (negated_genres & movie_genres):
                negation_violations += 1
                is_relevant = False
            if relevant_content_type and m.get("content_type") != relevant_content_type:
                content_type_violations += 1
                is_relevant = False

            if is_relevant:
                hits.append(m["serial_name"])

        graded_scores = None
        if use_llm:
            graded_scores = [
                float(llm_relevance_score(query, m)) for m in diversified
            ]

        metrics = evaluate_query(
            recommended_titles,
            hits,
            recommended_embeddings,
            recommended_genres=recommended_genres,
            genre_freq=self.genre_freq,
            graded_scores=graded_scores,
            k=k,
        )
        metrics["negation_violations"] = negation_violations
        metrics["content_type_violations"] = content_type_violations
        metrics["_recommended_titles"] = recommended_titles
        return metrics

    def _run_eval(self, test_set, k, weights, use_llm):
        self.stdout.write(
            f"Weights: semantic={weights['semantic']}, "
            f"metadata={weights['metadata']}, session={weights['session']}"
        )

        per_query = []
        all_recommendations = []
        for test_case in test_set:
            metrics = self._score_query(test_case, k, weights, use_llm)
            all_recommendations.append(metrics.pop("_recommended_titles", []))
            per_query.append(metrics)

            neg = metrics.get("negation_violations", 0)
            ct = metrics.get("content_type_violations", 0)
            violations = f" neg={neg}" if neg else ""
            violations += f" ct={ct}" if ct else ""
            llm_str = f" LLM={metrics['llm_relevance']:.1f}" if "llm_relevance" in metrics else ""
            self.stdout.write(
                f"  {test_case['query'][:50]:50s} P@{k}={metrics['precision_at_k']:.2f} "
                f"NDCG={metrics['ndcg_at_k']:.2f} "
                f"Div={metrics.get('diversity', 0):.2f} "
                f"Nov={metrics.get('novelty', 0):.2f}{llm_str}{violations}"
            )

        aggregated = aggregate_metrics(per_query)
        aggregated["coverage"] = coverage(all_recommendations, self.catalog_size)

        self.stdout.write("\n" + "=" * 70)
        self.stdout.write("AGGREGATE METRICS:")
        for key, value in aggregated.items():
            self.stdout.write(f"  {key}: {value:.4f}")

    def _run_sweep(self, test_set, k):
        use_llm = self._use_llm

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
                f"~{len(grid) * len(test_set) * k * 2}s estimated)..."
            )
        else:
            grid = [
                (sem, met, round(1.0 - sem - met, 2))
                for sem in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
                for met in [0.1, 0.2, 0.3, 0.4]
                if round(1.0 - sem - met, 2) >= 0
            ]
            self.stdout.write(f"Running weight grid search ({len(grid)} configs)...")

        results = []
        for i, (sem, met, ses) in enumerate(grid):
            weights = {"semantic": sem, "metadata": met, "session": ses}
            if use_llm:
                self.stdout.write(
                    f"  Config {i+1}/{len(grid)}: sem={sem} met={met} ses={ses}..."
                )

            per_query = []
            for test_case in test_set:
                metrics = self._score_query(test_case, k, weights, use_llm)
                metrics.pop("_recommended_titles", None)
                per_query.append(metrics)

            aggregated = aggregate_metrics(per_query)
            results.append((weights, aggregated))

        sort_key = "llm_relevance" if use_llm else "ndcg_at_k"
        results.sort(key=lambda x: x[1].get(sort_key, 0), reverse=True)

        header_llm = f" {'LLM':>6}" if use_llm else ""
        self.stdout.write(f"\nTop weight configurations by {sort_key}:")
        self.stdout.write(f"{'Semantic':>8} {'Meta':>6} {'Session':>8} {'P@k':>6} {'NDCG':>6} {'Nov':>6}{header_llm}")
        self.stdout.write("-" * (48 + (7 if use_llm else 0)))
        for weights, metrics in results[:len(grid)]:
            llm_col = f" {metrics.get('llm_relevance', 0):>6.2f}" if use_llm else ""
            self.stdout.write(
                f"{weights['semantic']:>8.2f} {weights['metadata']:>6.2f} "
                f"{weights['session']:>8.2f} "
                f"{metrics.get('precision_at_k', 0):>6.3f} "
                f"{metrics.get('ndcg_at_k', 0):>6.3f} "
                f"{metrics.get('novelty', 0):>6.3f}{llm_col}"
            )
