import pytest

from django.conf import settings

from unittest.mock import patch

from core.scoring import (
    SIGNALS,
    _metadata_score,
    _semantic_score,
    effective_weights,
    normalize_scores,
    mmr_diversify,
    score_candidates,
    weighted_rating,
)


@pytest.fixture(autouse=True)
def _catalog_mean():
    """The catalog mean comes from the DB; these tests run on the host without
    one. Tests that need a specific value (or None) patch it again."""
    with patch("core.scoring.catalog_mean_rating", return_value=6.3):
        yield


WEIGHTS = {"semantic": 0.5, "metadata": 0.3, "popularity": 0.2, "quality": 0.0}


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
        assert normalize_scores([0.5, 0.75, 1.0]) == pytest.approx([0.0, 0.5, 1.0])

    def test_constant_signal_collapses_to_neutral(self):
        assert normalize_scores([0.6, 0.6, 0.6]) == [settings.NEUTRAL_SCORE] * 3

    def test_empty_input(self):
        assert normalize_scores([]) == []

    def test_single_value_is_neutral(self):
        assert normalize_scores([0.42]) == [settings.NEUTRAL_SCORE]


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
            key in result[0] for key in ("total", "semantic", "metadata", "popularity", "quality", "raw_scores")
        )

    def test_session_is_not_a_signal(self):
        candidates = [
            _make_movie(embedding=[1.0, 0.0, 0.0], movie_id=1),
            _make_movie(embedding=[0.0, 1.0, 0.0], movie_id=2),
        ]
        result = score_candidates(candidates, [1.0, 0.0, 0.0], {"genres": []}, weights=WEIGHTS)
        assert "session" not in result[0]
        assert set(result[0]["raw_scores"]) == {"semantic", "metadata", "popularity", "quality"}

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
        """A signal that ties across the pool scores neutral, so ordering is
        decided by the signals that actually vary."""
        candidates = [
            _make_movie(embedding=[1.0, 0.0, 0.0], movie_id=1),
            _make_movie(embedding=[0.0, 1.0, 0.0], movie_id=2),
        ]
        result = score_candidates(
            candidates, [1.0, 0.0, 0.0], {"genres": []}, weights=WEIGHTS
        )
        assert all(r["popularity"] == settings.NEUTRAL_SCORE for r in result)
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


class TestWeightedRating:
    def test_weighted_rating_shrinks_low_vote_movies_to_catalog_mean(self):
        assert weighted_rating(9.0, 10, 500, 6.3) == pytest.approx(6.3, abs=0.06)
        assert weighted_rating(8.0, 50000, 500, 6.3) == pytest.approx(8.0, abs=0.02)

    def test_weighted_rating_zero_votes_is_catalog_mean(self):
        assert weighted_rating(0.0, 0, 500, 6.3) == 6.3


def _rated(movie_id, vote_average, vote_count, embedding):
    return {
        **_make_movie(embedding=embedding, movie_id=movie_id),
        "vote_average": vote_average,
        "vote_count": vote_count,
    }


class TestQualitySignal:
    def test_signals_include_quality(self):
        assert SIGNALS == ("semantic", "metadata", "popularity", "quality")

    def test_quality_signal_neutral_without_catalog_mean(self):
        candidates = [
            _rated(1, 9.0, 20000, [1.0, 0.0, 0.0]),
            _rated(2, 3.0, 20000, [0.0, 1.0, 0.0]),
        ]
        with patch("core.scoring.catalog_mean_rating", return_value=None):
            result = score_candidates(candidates, [1.0, 0.0, 0.0], {"genres": []}, weights=WEIGHTS)
        assert all(r["quality"] == settings.NEUTRAL_SCORE for r in result)

    def test_quality_signal_ranks_higher_rated_movie_first(self):
        candidates = [
            _rated(1, 5.0, 20000, [1.0, 0.0, 0.0]),
            _rated(2, 8.5, 20000, [1.0, 0.0, 0.0]),
        ]
        result = score_candidates(candidates, [1.0, 0.0, 0.0], {"genres": []}, weights=WEIGHTS)
        by_id = {r["id"]: r for r in result}
        assert by_id[2]["quality"] == pytest.approx(1.0)
        assert by_id[1]["quality"] == pytest.approx(0.0)
        assert by_id[2]["raw_scores"]["quality"] > by_id[1]["raw_scores"]["quality"]

    def test_missing_vote_fields_score_as_catalog_mean(self):
        candidates = [
            _make_movie(embedding=[1.0, 0.0, 0.0], movie_id=1),
            _rated(2, 9.0, 20000, [1.0, 0.0, 0.0]),
        ]
        result = score_candidates(candidates, [1.0, 0.0, 0.0], {"genres": []}, weights=WEIGHTS)
        assert result[0]["raw_scores"]["quality"] == pytest.approx(6.3)

    def test_quality_weight_zero_keeps_order(self):
        """With the shipped weights, vote data does not change the ordering."""
        assert settings.SCORE_WEIGHTS["quality"] == 0.0
        embeddings = [[1.0, 0.0, 0.0], [0.9, 0.4, 0.0], [0.3, 0.9, 0.0], [0.0, 1.0, 0.0]]
        pool = [
            {**_make_movie(embedding=e, movie_id=i), "popularity": 10.0 * (4 - i)}
            for i, e in enumerate(embeddings)
        ]
        rated = [
            {**m, "vote_average": va, "vote_count": vc}
            for m, (va, vc) in zip(pool, [(4.0, 90000), (9.0, 90000), (5.0, 3), (9.5, 90000)])
        ]
        before = score_candidates(pool, [1.0, 0.0, 0.0], {"genres": []})
        after = score_candidates(rated, [1.0, 0.0, 0.0], {"genres": []})
        order = lambda rs: [r["id"] for r in sorted(rs, key=lambda r: -r["total"])]
        assert order(before) == order(after)
        assert [r["total"] for r in before] == pytest.approx([r["total"] for r in after])


class TestEffectiveWeights:
    FULL = {"semantic": 0.4, "metadata": 0.3, "popularity": 0.1, "quality": 0.2}

    def test_effective_weights_drop_quality_for_obscure(self):
        result = effective_weights(self.FULL, {"wants_obscure": True})
        assert result["quality"] == 0
        assert sum(result.values()) == pytest.approx(1.0)
        assert result["semantic"] / result["metadata"] == pytest.approx(4 / 3)
        assert result["semantic"] / result["popularity"] == pytest.approx(4.0)

    def test_unchanged_without_the_flag(self):
        assert effective_weights(self.FULL, {}) == self.FULL
        assert effective_weights(self.FULL, {"wants_obscure": False}) == self.FULL

    def test_does_not_mutate_input(self):
        effective_weights(self.FULL, {"wants_obscure": True})
        assert self.FULL["quality"] == 0.2

    def test_obscure_with_zero_quality_weight_is_unchanged(self):
        assert effective_weights(WEIGHTS, {"wants_obscure": True}) == WEIGHTS

    def test_score_candidates_applies_it(self):
        candidates = [
            _rated(1, 9.0, 20000, [0.0, 1.0, 0.0]),
            _rated(2, 3.0, 20000, [1.0, 0.0, 0.0]),
        ]
        weights = {"semantic": 0.3, "metadata": 0.0, "popularity": 0.0, "quality": 0.7}
        plain = score_candidates(candidates, [1.0, 0.0, 0.0], {"genres": []}, weights=weights)
        obscure = score_candidates(
            candidates, [1.0, 0.0, 0.0], {"genres": [], "wants_obscure": True}, weights=weights
        )
        assert max(plain, key=lambda r: r["total"])["id"] == 1
        assert max(obscure, key=lambda r: r["total"])["id"] == 2


@pytest.mark.parametrize("rating", [0.0, None])
def test_missing_rating_with_positive_votes_uses_catalog_mean(rating):
    from core.scoring import _quality_score

    assert _quality_score({"vote_average": rating, "vote_count": 1000}, 6.3) == pytest.approx(6.3)


def test_obscure_preserves_reference_and_rating_constraints():
    intent = {"wants_obscure": True, "reference_films": ["Alien"], "min_vote_average": 7.0}
    weights = {"semantic": 0.4, "metadata": 0.3, "popularity": 0.1, "quality": 0.2}
    assert effective_weights(weights, intent) == pytest.approx(
        {"semantic": 0.5, "metadata": 0.375, "popularity": 0.125, "quality": 0.0}
    )
    assert intent == {"wants_obscure": True, "reference_films": ["Alien"], "min_vote_average": 7.0}
