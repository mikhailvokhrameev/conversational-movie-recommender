"""Tests for core.pipeline.generate_and_score -- the stage wiring only; each
stage has its own tests. No DB: every stage is patched."""

from unittest.mock import patch

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
            top, reranked = generate_and_score([0.1], {"genres": ["Drama"]}, [0.2], query_text="q")

        assert top == pool[:2]
        assert reranked == pool
        g.assert_called_once_with([0.1], {"genres": ["Drama"]}, query_text="q")
        s.assert_called_once_with(pool, [0.1], {"genres": ["Drama"]}, [0.2], weights=None)
        r.assert_called_once_with("q", pool)
        m.assert_called_once_with(pool, top_n=2)

    def test_eval_overrides_for_weights_and_top_n(self):
        pool = [{"id": i, "total": 0.5} for i in range(5)]
        weights = {"semantic": 1.0, "metadata": 0.0, "session": 0.0, "popularity": 0.0}
        gen, score, rerank, mmr = _stages(pool)
        with gen, score as s, rerank, mmr as m:
            top, _ = generate_and_score([0.1], {}, None, query_text="q", weights=weights, top_n=3)

        assert len(top) == 3
        assert s.call_args.kwargs["weights"] == weights
        m.assert_called_once_with(pool, top_n=3)
