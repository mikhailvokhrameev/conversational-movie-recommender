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


class TestPinnedRevision:
    def test_model_is_loaded_at_the_configured_revision(self, settings):
        from unittest.mock import patch

        from core import embedding_service

        settings.EMBEDDING_MODEL = "org/embedder"
        settings.EMBEDDING_REVISION = "abc123"
        with patch("sentence_transformers.SentenceTransformer") as ctor:
            adapter = embedding_service.LocalEmbeddingAdapter()
            assert adapter.get_model() is ctor.return_value
            assert adapter.get_model() is ctor.return_value
        ctor.assert_called_once_with("org/embedder", revision="abc123")



class TestEmbeddingAdapters:
    @pytest.fixture
    def remote(self, settings):
        from copy import deepcopy
        from core import embedding_service as service
        from core.inference_config import resolve_inference_config
        settings.EFFECTIVE_PARAMS = resolve_inference_config(deepcopy(settings.EFFECTIVE_PARAMS), 'openrouter')
        settings.EMBEDDING_BATCH_SIZE = 2
        return service

    @staticmethod
    def response(indices):
        return {'data': [{'index': i, 'embedding': [float(i + 1)] * 1024} for i in indices]}

    def test_reorders_and_splits_batches(self, remote):
        from unittest.mock import patch
        with patch('core.openrouter_client.post_json', side_effect=[self.response([1, 0]), self.response([0])]) as post:
            result = remote.OpenRouterEmbeddingAdapter().encode_texts(['a', 'b', 'c'])
        assert result.shape == (3, 1024)
        assert result[:, 0].tolist() == [1, 2, 1]
        assert [call.args[1]['input'] for call in post.call_args_list] == [['a', 'b'], ['c']]
        payload = post.call_args_list[0].args[1]
        assert post.call_args_list[0].args[0] == '/embeddings'
        assert payload['model'] == 'baai/bge-m3'
        assert payload['provider'] == {'order': ['DeepInfra'], 'allow_fallbacks': False}
        assert post.call_args_list[0].kwargs['timeout'] == 60

    @pytest.mark.parametrize('kind', ['missing', 'duplicate', 'range', 'bool_index', 'nan', 'inf', 'dimension', 'zero', 'string', 'malformed'])
    def test_rejects_invalid_response(self, remote, kind):
        from unittest.mock import patch
        from core.openrouter_client import InferenceError
        body = self.response([0, 1])
        if kind == 'missing':
            body['data'].pop()
        elif kind == 'duplicate':
            body['data'][1]['index'] = 0
        elif kind == 'range':
            body['data'][1]['index'] = 2
        elif kind == 'bool_index':
            body['data'][1]['index'] = True
        elif kind == 'malformed':
            body = {'data': ['private upstream content']}
        else:
            body['data'][0]['embedding'] = {'nan': [float('nan')] * 1024, 'inf': [float('inf')] * 1024,
                'dimension': [1.0] * 1023, 'zero': [0.0] * 1024, 'string': ['1'] * 1024}[kind]
        with patch('core.openrouter_client.post_json', return_value=body):
            with pytest.raises(InferenceError, match='embedding_response') as exc:
                remote.OpenRouterEmbeddingAdapter().encode_texts(['a', 'b'])
        assert exc.value.__context__ is None

    def test_empty_input_does_no_work(self, remote):
        from unittest.mock import patch
        with patch('core.openrouter_client.post_json') as post:
            assert remote.OpenRouterEmbeddingAdapter().encode_texts([]).shape == (0, 1024)
            assert remote.LocalEmbeddingAdapter().encode_texts([]).shape == (0, 1024)
        post.assert_not_called()

    def test_service_uses_assembled_backend_for_query_and_batch(self, remote):
        from unittest.mock import Mock, patch
        import numpy as np
        backend = Mock()
        backend.embedder.encode_texts.return_value = np.ones((1, 1024))
        with patch('core.inference_backends.get_inference_backends', return_value=backend):
            assert remote.encode_query('unchanged query') == [1.] * 1024
            remote.encode_texts(['unchanged text'])
        assert backend.embedder.encode_texts.call_args_list[0].args == (['unchanged query'],)
        assert backend.embedder.encode_texts.call_args_list[1].args == (['unchanged text'],)

    def test_local_preserves_native_vectors(self, remote):
        from unittest.mock import Mock, patch
        import numpy as np
        adapter = remote.LocalEmbeddingAdapter()
        vectors = np.full((2, 1024), 3.)
        model = Mock()
        model.encode.return_value = vectors
        with patch.object(adapter, 'get_model', return_value=model):
            assert adapter.encode_texts(['a', 'b']) is vectors
        model.encode.assert_called_once_with(['a', 'b'], convert_to_numpy=True)

    def test_cloud_assembly_never_imports_local_models(self, remote, monkeypatch):
        import builtins
        from unittest.mock import patch
        from core.inference_backends import _build_backends
        original = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name == 'sentence_transformers' or name.startswith('sentence_transformers.'):
                raise AssertionError('cloud must not import local models')
            return original(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', guarded)
        with patch('core.openrouter_client.post_json', return_value=self.response([0])):
            assert _build_backends().embedder.encode_texts(['query']).shape == (1, 1024)
