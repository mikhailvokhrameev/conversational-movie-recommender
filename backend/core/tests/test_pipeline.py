"""Tests for core.pipeline.generate_and_score -- the stage wiring only; each
stage has its own tests. No DB: every stage is patched."""

from unittest.mock import MagicMock, patch

import pytest

from core.pipeline import generate_and_score


def _stages(pool):
    return (
        patch("core.pipeline.generate_candidates", return_value=pool),
        patch("core.pipeline.score_candidates", side_effect=lambda c, *a, **kw: c),
        patch("core.pipeline.rerank_candidates", side_effect=lambda q, c: c),
        patch("core.pipeline.mmr_diversify", side_effect=lambda c, top_n: c[:top_n]),
    )


class TestGenerateAndScore:
    def test_runs_stages_in_order_and_returns_top_and_pool(self, settings):
        settings.TOP_N = 2
        pool = [{"id": i, "total": 1.0 - i / 10} for i in range(4)]
        gen, score, rerank, mmr = _stages(pool)
        with gen as g, score as s, rerank as r, mmr as m:
            top, reranked = generate_and_score([0.1], {"genres": ["Drama"]}, query_text="q")

        assert top == pool[:2]
        assert reranked == pool
        g.assert_called_once_with([0.1], {"genres": ["Drama"]}, query_text="q")
        s.assert_called_once_with(pool, [0.1], {"genres": ["Drama"]}, weights=None)
        r.assert_called_once_with("q", pool)
        m.assert_called_once_with(pool, top_n=2)

    def test_eval_overrides_for_weights_and_top_n(self):
        pool = [{"id": i, "total": 0.5} for i in range(5)]
        weights = {"semantic": 1.0, "metadata": 0.0, "popularity": 0.0}
        gen, score, rerank, mmr = _stages(pool)
        with gen, score as s, rerank, mmr as m:
            top, _ = generate_and_score([0.1], {}, query_text="q", weights=weights, top_n=3)

        assert len(top) == 3
        assert s.call_args.kwargs["weights"] == weights
        m.assert_called_once_with(pool, top_n=3)


def _scored(id_, rerank_score):
    return {"id": id_, "total": 0.5, "rerank_score": rerank_score}


class TestRelevanceGate:
    """`reranking.min_score`: drop candidates the cross-encoder scored below it
    (and the ones it never scored) before diversification."""

    def _run(self, pool, top_n=5):
        gen = patch("core.pipeline.generate_candidates", return_value=pool)
        score = patch("core.pipeline.score_candidates", side_effect=lambda c, *a, **kw: c)
        rerank = patch("core.pipeline.rerank_candidates", side_effect=lambda q, c: c)
        mmr = patch("core.pipeline.mmr_diversify", side_effect=lambda c, top_n: c[:top_n])
        with gen, score, rerank, mmr as m:
            top, reranked = generate_and_score([0.1], {}, query_text="q", top_n=top_n)
        return top, reranked, m

    def test_off_by_default_leaves_output_unchanged(self, settings):
        settings.RERANK_MIN_SCORE = None
        pool = [_scored(1, 0.01), _scored(2, 0.9), {"id": 3, "total": 0.1}]
        top, reranked, _ = self._run(pool)
        assert top == pool
        assert reranked == pool

    def test_drops_below_threshold_and_unscored_before_mmr(self, settings):
        settings.RERANK_MIN_SCORE = 0.3
        pool = [_scored(1, 0.9), _scored(2, 0.29), _scored(3, 0.3), {"id": 4, "total": 0.1}]
        top, reranked, mmr = self._run(pool)
        assert [m["id"] for m in top] == [1, 3]
        # MMR saw only the gated list, so a dropped candidate cannot be picked.
        assert [c["id"] for c in mmr.call_args.args[0]] == [1, 3]
        # The returned pool stays the full reranked pool (hallucination check).
        assert reranked == pool

    def test_all_below_threshold_gives_empty_top(self, settings):
        settings.RERANK_MIN_SCORE = 0.5
        top, reranked, _ = self._run([_scored(1, 0.1), _scored(2, 0.2)])
        assert top == []
        assert len(reranked) == 2

    def test_gate_skipped_with_warning_when_reranker_did_not_score(self, settings, caplog):
        settings.RERANK_MIN_SCORE = 0.5
        pool = [{"id": 1, "total": 0.9}, {"id": 2, "total": 0.8}]  # no rerank_score
        with caplog.at_level("WARNING", logger="core.pipeline"):
            top, _, _ = self._run(pool)
        assert top == pool
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        assert "gate skipped" in warnings[0].getMessage()

    def test_gate_skipped_when_model_unavailable(self, settings, caplog):
        """Through the real rerank_candidates: get_model() is None -> the
        pool comes back unscored -> the gate must fail open."""
        settings.RERANK_MIN_SCORE = 0.5
        settings.RERANKER_ENABLED = True
        settings.RERANK_TOP_K = 10
        pool = [{"id": 1, "total": 0.9}, {"id": 2, "total": 0.8}]
        with patch("core.pipeline.generate_candidates", return_value=pool), \
             patch("core.pipeline.score_candidates", side_effect=lambda c, *a, **kw: c), \
             patch("core.reranking.LocalRerankingAdapter.get_model", return_value=None), \
             patch("core.pipeline.mmr_diversify", side_effect=lambda c, top_n: c[:top_n]), \
             caplog.at_level("WARNING", logger="core.pipeline"):
            top, _ = generate_and_score([0.1], {}, query_text="q", top_n=5)
        assert [m["id"] for m in top] == [1, 2]
        assert any("gate skipped" in r.getMessage() for r in caplog.records)

    def test_no_warning_when_there_is_nothing_to_gate(self, settings, caplog):
        settings.RERANK_MIN_SCORE = 0.5
        with caplog.at_level("WARNING", logger="core.pipeline"):
            top, _, _ = self._run([])
        assert top == []
        assert not caplog.records

    def test_apply_gate_false_bypasses_it_for_calibration(self, settings):
        settings.RERANK_MIN_SCORE = 0.5
        pool = [_scored(1, 0.1)]
        with patch("core.pipeline.generate_candidates", return_value=pool), \
             patch("core.pipeline.score_candidates", side_effect=lambda c, *a, **kw: c), \
             patch("core.pipeline.rerank_candidates", side_effect=lambda q, c: c), \
             patch("core.pipeline.mmr_diversify", side_effect=lambda c, top_n: c[:top_n]):
            top, _ = generate_and_score([0.1], {}, query_text="q", apply_gate=False)
        assert top == pool


def _film(id_, title, original=None):
    return {"id": id_, "tmdb_id": id_, "serial_name": title,
            "original_title": original or title, "total": 0.5}


class TestExclusion:
    def _run(self, pool, intent=None, top_n=3, **kwargs):
        gen, score, rerank, mmr = _stages(pool)
        with gen, score, rerank, mmr:
            return generate_and_score(
                [0.1], intent or {}, query_text="q", top_n=top_n, **kwargs
            )

    def test_reference_film_is_excluded(self):
        pool = [_film(1, "The Matrix"), _film(2, "Dark City")]
        top, _ = self._run(pool, {"reference_films": ["the matrix"]})
        assert [c["tmdb_id"] for c in top] == [2]

    def test_reference_film_matched_by_original_title(self):
        pool = [_film(1, "Amelie", original="Le Fabuleux Destin d'Amélie Poulain"),
                _film(2, "Dark City")]
        top, _ = self._run(pool, {"reference_films": ["Le fabuleux destin d'Amélie Poulain"]})
        assert [c["tmdb_id"] for c in top] == [2]

    def test_similar_title_is_kept(self):
        pool = [_film(1, "Sherlock Holmes"), _film(2, "Sherlock Holmes: A Game of Shadows")]
        top, reranked = self._run(pool, {"reference_films": ["Sherlock Holmes"]})
        assert [c["tmdb_id"] for c in top] == [2]
        assert len(reranked) == 2

    def test_exclude_ids_are_dropped(self):
        pool = [_film(1, "A One"), _film(2, "B Two"), _film(3, "C Three")]
        top, _ = self._run(pool, exclude_ids={1, 3})
        assert [c["tmdb_id"] for c in top] == [2]

    def test_exclusion_runs_before_mmr(self):
        pool = [_film(i, f"Film {i}") for i in range(1, 6)]
        top, _ = self._run(pool, exclude_ids={1, 2}, top_n=3)
        assert [c["tmdb_id"] for c in top] == [3, 4, 5]


@pytest.mark.parametrize("provider", ["local", "openrouter"])
def test_shared_pipeline_adapter_contract(provider, settings):
    from types import SimpleNamespace
    from core.reranking import LocalRerankingAdapter, OpenRouterRerankingAdapter

    settings.RERANKER_ENABLED = True
    settings.RERANK_TOP_K = 3
    settings.RERANK_WEIGHT = 1.0
    settings.RERANK_MIN_SCORE = None
    pool = [dict(_film(i, f"Film {i}"), total=1 - i / 10,
                 embedding=[1.0, 0.0]) for i in range(1, 5)]
    adapter = {"local": LocalRerankingAdapter, "openrouter": OpenRouterRerankingAdapter}[provider]()
    model = MagicMock()
    model.predict.return_value = [.1, .9, .5]
    body = {'results': [{'index': i, 'relevance_score': s} for i, s in [(2, .5), (0, .1), (1, .9)]]}
    with patch("core.pipeline.generate_candidates", return_value=pool), \
         patch("core.pipeline.score_candidates", side_effect=lambda c, *a, **kw: c), \
         patch("core.reranking.get_inference_backends", return_value=SimpleNamespace(reranker=adapter)), \
         patch.object(LocalRerankingAdapter, "get_model", return_value=model), \
         patch("core.openrouter_client.post_json", return_value=body):
        top, reranked = generate_and_score([1.0, 0.0], {}, query_text="q", top_n=2)
    assert [c['id'] for c in reranked] == [2, 3, 1]
    assert [c['rerank_score'] for c in reranked] == [.9, .5, .1]
    assert [c['id'] for c in top] == [2, 3]


@pytest.mark.parametrize('failure', ['http', 'incomplete'])
def test_cloud_reranking_failure_keeps_full_pool_and_gate_open(failure, settings, caplog):
    from types import SimpleNamespace
    from core.reranking import OpenRouterRerankingAdapter
    from core.openrouter_client import InferenceError

    settings.RERANKER_ENABLED = True
    settings.RERANK_TOP_K = 2
    settings.RERANK_MIN_SCORE = .5
    pool = [dict(_film(i, f'Film {i}'), total=i / 10) for i in range(1, 4)]
    kwargs = {'side_effect': InferenceError('http', 503)} if failure == 'http' else {'return_value': {'results': [{'index': 0, 'relevance_score': .9}]}}
    with patch('core.pipeline.generate_candidates', return_value=pool), \
         patch('core.pipeline.score_candidates', side_effect=lambda c, *a, **kw: c), \
         patch('core.reranking.get_inference_backends', return_value=SimpleNamespace(reranker=OpenRouterRerankingAdapter())), \
         patch('core.openrouter_client.post_json', **kwargs), \
         patch('core.pipeline.mmr_diversify', side_effect=lambda c, top_n: c[:top_n]):
        top, reranked = generate_and_score([.1], {}, query_text='q', top_n=5)
    assert [c['id'] for c in top] == [3, 2, 1]
    assert top == reranked
    assert all('rerank_score' not in c for c in reranked)
    assert any('gate skipped' in record.getMessage() for record in caplog.records)
