import pytest

from core.embedding_service import cosine_similarity


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity([0, 0, 0], [1, 0, 0]) == 0.0

    def test_non_unit_vectors(self):
        sim = cosine_similarity([3, 4], [6, 8])
        assert sim == pytest.approx(1.0, abs=1e-6)


def _legacy_text(name, genres, description, keywords):
    """The f-string generate_embeddings used before the template moved into
    embedding_service, with its literals inlined. Existing Movie.embedding
    vectors were built from exactly this, so the shared builder must match it."""
    genre_str = ", ".join(genres) if genres else ""
    desc_truncated = (description or "")[:500]
    keyword_str = ", ".join((keywords or [])[:10])
    return f"{name}. {genre_str}. {desc_truncated}. Keywords: {keyword_str}"


class TestBuildMovieEmbeddingText:
    @pytest.mark.parametrize(
        "genres, description, keywords",
        [
            (["Drama", "Crime"], "A short plot.", ["heist", "noir"]),
            ([], "A short plot.", ["heist"]),
            (None, "A short plot.", ["heist"]),
            (["Drama"], None, ["heist"]),
            (["Drama"], "", []),
            (["Drama"], "A short plot.", None),
            (["Drama"], "x" * 499, ["k"]),
            (["Drama"], "x" * 500, ["k"]),
            (["Drama"], "y" * 501, ["k"]),
            (["Drama"], "z" * 5000, ["k"]),
            (["Drama"], "Кириллица " * 100, [f"kw{i}" for i in range(9)]),
            (["Drama"], "plot", [f"kw{i}" for i in range(10)]),
            (["Drama"], "plot", [f"kw{i}" for i in range(11)]),
            (["Drama"], "plot", [f"kw{i}" for i in range(40)]),
        ],
    )
    def test_matches_the_legacy_template_byte_for_byte(self, genres, description, keywords):
        from core.embedding_service import build_movie_embedding_text

        built = build_movie_embedding_text("Heat", genres, description, keywords)

        assert built == _legacy_text("Heat", genres, description, keywords)

    def test_truncates_description_and_keywords_to_the_configured_limits(self):
        from core.embedding_service import build_movie_embedding_text

        built = build_movie_embedding_text(
            "T", [], "abcdef", ["a", "b", "c", "d"], description_chars=3, max_keywords=2
        )

        assert built == "T. . abc. Keywords: a, b"

    def test_defaults_come_from_settings(self):
        from django.conf import settings

        assert settings.EMBEDDING_TEXT_DESCRIPTION_CHARS == 500
        assert settings.EMBEDDING_TEXT_MAX_KEYWORDS == 10

    def test_text_version_is_a_positive_int(self):
        from core.embedding_service import EMBEDDING_TEXT_VERSION

        assert isinstance(EMBEDDING_TEXT_VERSION, int) and EMBEDDING_TEXT_VERSION >= 1


class TestEmbeddingTextSettingsValidation:
    def test_accepts_positive_ints(self):
        from recommender.settings import _validate_embedding_text

        _validate_embedding_text({"text": {"description_chars": 1, "max_keywords": 1}})

    @pytest.mark.parametrize("bad", [0, -1, 2.5, "500", True, None, [500]])
    @pytest.mark.parametrize("key", ["description_chars", "max_keywords"])
    def test_rejects_non_positive_or_non_int(self, key, bad):
        from recommender.settings import _validate_embedding_text

        text = {"description_chars": 500, "max_keywords": 10, key: bad}
        with pytest.raises(ValueError, match=f"embedding.text.{key}"):
            _validate_embedding_text({"text": text})

    def test_rejects_missing_text_block(self):
        from recommender.settings import _validate_embedding_text

        with pytest.raises(ValueError, match="embedding.text"):
            _validate_embedding_text({})
