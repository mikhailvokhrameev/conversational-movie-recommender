import pytest

from core.scoring import (
    NEUTRAL_SCORE,
    _metadata_score,
    _normalize,
    _semantic_score,
    mmr_diversify,
    score_candidates,
)


WEIGHTS = {"semantic": 0.4, "metadata": 0.3, "session": 0.3}


def _make_movie(embedding=None, genres=None, movie_id=1):
    return {
        "id": movie_id,
        "serial_name": "Test",
        "genres": genres or [],
        "embedding": embedding,
    }


class TestSemanticScore:
    def test_identical_embedding_gives_max_semantic(self):
        emb = [1.0, 0.0, 0.0]
        assert _semantic_score(_make_movie(embedding=emb), emb) == pytest.approx(1.0)

    def test_no_embedding_gives_zero_semantic(self):
        assert _semantic_score(_make_movie(embedding=None), [1.0, 0.0, 0.0]) == 0.0


class TestNormalize:
    def test_maps_spread_to_unit_range(self):
        assert _normalize([0.5, 0.75, 1.0]) == pytest.approx([0.0, 0.5, 1.0])

    def test_constant_signal_collapses_to_neutral(self):
        assert _normalize([0.6, 0.6, 0.6]) == [NEUTRAL_SCORE] * 3

    def test_empty_input(self):
        assert _normalize([]) == []

    def test_single_value_is_neutral(self):
        assert _normalize([0.42]) == [NEUTRAL_SCORE]


class TestScoreCandidates:
    def test_empty_candidates(self):
        assert score_candidates([], [1.0, 0.0], {"genres": []}, weights=WEIGHTS) == []

    def test_returns_all_score_components(self):
        candidates = [
            _make_movie(embedding=[1.0, 0.0, 0.0], movie_id=1),
            _make_movie(embedding=[0.0, 1.0, 0.0], movie_id=2),
        ]
        result = score_candidates(candidates, [1.0, 0.0, 0.0], {"genres": []}, weights=WEIGHTS)
        assert all(
            key in result[0] for key in ("total", "semantic", "metadata", "session", "raw_scores")
        )

    def test_does_not_mutate_input_candidates(self):
        candidates = [_make_movie(embedding=[1.0, 0.0, 0.0])]
        score_candidates(candidates, [1.0, 0.0, 0.0], {"genres": []}, weights=WEIGHTS)
        assert "total" not in candidates[0]

    def test_normalization_spans_unit_range_across_set(self):
        """The best and worst candidate on a signal anchor 1.0 and 0.0."""
        candidates = [
            _make_movie(embedding=[1.0, 0.0, 0.0], movie_id=1),
            _make_movie(embedding=[-1.0, 0.0, 0.0], movie_id=2),
            _make_movie(embedding=[0.0, 1.0, 0.0], movie_id=3),
        ]
        result = score_candidates(candidates, [1.0, 0.0, 0.0], {"genres": []}, weights=WEIGHTS)
        semantic = [r["semantic"] for r in result]
        assert max(semantic) == pytest.approx(1.0)
        assert min(semantic) == pytest.approx(0.0)

    def test_raw_scores_preserved_unnormalized(self):
        candidates = [
            _make_movie(embedding=[1.0, 0.0, 0.0], movie_id=1),
            _make_movie(embedding=[-1.0, 0.0, 0.0], movie_id=2),
        ]
        result = score_candidates(candidates, [1.0, 0.0, 0.0], {"genres": []}, weights=WEIGHTS)
        assert result[0]["raw_scores"]["semantic"] == pytest.approx(1.0)
        assert result[1]["raw_scores"]["semantic"] == pytest.approx(0.0)

    def test_uninformative_signal_does_not_change_ranking(self):
        """With no session vector every session score ties, so ordering is
        decided by the signals that actually vary."""
        candidates = [
            _make_movie(embedding=[1.0, 0.0, 0.0], movie_id=1),
            _make_movie(embedding=[0.0, 1.0, 0.0], movie_id=2),
        ]
        result = score_candidates(
            candidates, [1.0, 0.0, 0.0], {"genres": []}, session_vector=None, weights=WEIGHTS
        )
        assert all(r["session"] == NEUTRAL_SCORE for r in result)
        assert result[0]["total"] > result[1]["total"]

    def test_metadata_no_longer_dominates_semantic(self):
        """Regression test for the un-normalized fusion bug.

        Movie A is a far better semantic match; movie B wins only on genre
        overlap. With semantic weighted 0.4 vs metadata 0.3, A must rank
        first. Before normalization B won, because raw metadata spanned
        {0.0, 1.0} while raw semantic was squeezed into a narrow band.
        """
        intent = {"genres": ["Комедии"]}
        movie_a = _make_movie(embedding=[1.0, 0.0, 0.0], genres=["Драмы"], movie_id=1)
        movie_b = _make_movie(embedding=[0.88, 0.47, 0.0], genres=["Комедии"], movie_id=2)

        result = score_candidates([movie_a, movie_b], [1.0, 0.0, 0.0], intent, weights=WEIGHTS)
        by_id = {r["id"]: r for r in result}
        assert by_id[1]["total"] > by_id[2]["total"]


class TestMetadataScore:
    def test_full_overlap(self):
        assert _metadata_score({"genres": ["Комедии"]}, {"genres": ["Комедии"]}) == 1.0

    def test_no_overlap(self):
        assert _metadata_score({"genres": ["Драмы"]}, {"genres": ["Комедии"]}) == 0.0

    def test_partial_overlap(self):
        score = _metadata_score({"genres": ["Комедии", "Драмы"]}, {"genres": ["Комедии", "Триллеры"]})
        assert score == pytest.approx(0.5)

    def test_empty_intent_genres_returns_default(self):
        assert _metadata_score({"genres": ["Комедии"]}, {"genres": []}) == 0.5


class TestMMRDiversify:
    def test_returns_all_when_fewer_than_top_n(self):
        candidates = [{"total": 0.9, "embedding": [1, 0]}, {"total": 0.8, "embedding": [0, 1]}]
        result = mmr_diversify(candidates, top_n=5)
        assert len(result) == 2

    def test_selects_top_n(self):
        candidates = [
            {"total": 0.9, "embedding": [1, 0, 0]},
            {"total": 0.8, "embedding": [0, 1, 0]},
            {"total": 0.7, "embedding": [0, 0, 1]},
            {"total": 0.6, "embedding": [1, 1, 0]},
        ]
        result = mmr_diversify(candidates, top_n=2)
        assert len(result) == 2

    def test_highest_score_always_first(self):
        candidates = [
            {"total": 0.5, "embedding": [1, 0]},
            {"total": 0.9, "embedding": [0, 1]},
            {"total": 0.7, "embedding": [1, 1]},
        ]
        result = mmr_diversify(candidates, top_n=2)
        assert result[0]["total"] == 0.9

    def test_diversity_prefers_different_embeddings(self):
        candidates = [
            {"total": 0.9, "embedding": [1, 0, 0]},
            {"total": 0.85, "embedding": [0.99, 0.01, 0]},
            {"total": 0.8, "embedding": [0, 0, 1]},
        ]
        result = mmr_diversify(candidates, top_n=2, lambda_param=0.5)
        assert result[1]["total"] == 0.8
