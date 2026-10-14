"""generate_embeddings / check_embedding_index against the EmbeddingIndex record.

These need Postgres (pgvector columns); the decisions they act on are covered
without a DB in core/tests/test_embedding_provenance.py. The encoder is faked
at the command module so no model is downloaded."""

from io import StringIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from django.conf import settings
from django.core.management import call_command
from django.core.management.base import CommandError

from core.embedding_provenance import current_provenance
from movies.models import EmbeddingIndex, Movie

GENERATE = "movies.management.commands.generate_embeddings"


def _vec(value):
    return [value] * settings.EMBEDDING_DIMENSIONS


def _movie(tmdb_id, embedding=None):
    return Movie.objects.create(
        tmdb_id=tmdb_id, serial_name=f"Film {tmdb_id}", genres=["Drama"],
        description="plot", keywords=["k"], embedding=embedding,
    )


def _generate(*args):
    """Run generate_embeddings with a fake encoder returning all-ones vectors."""
    with patch(f"{GENERATE}.SentenceTransformer") as encoder_cls:
        encoder = encoder_cls.return_value
        encoder.encode.side_effect = lambda texts, **_: np.ones((len(texts), settings.EMBEDDING_DIMENSIONS))
        out = StringIO()
        call_command("generate_embeddings", *args, stdout=out)
    return encoder_cls, encoder, out.getvalue()


def _record(**changes):
    EmbeddingIndex.record({**current_provenance(), **changes})


@pytest.mark.django_db
class TestEmbeddingIndexModel:
    def test_stored_is_none_without_a_record(self):
        assert EmbeddingIndex.stored() is None

    def test_record_round_trips_current_provenance(self):
        EmbeddingIndex.record(current_provenance())

        assert EmbeddingIndex.stored() == current_provenance()

    def test_unpinned_revision_round_trips_as_null(self):
        EmbeddingIndex.record({**current_provenance(), "revision": None})

        assert EmbeddingIndex.stored()["revision"] is None

    def test_record_overwrites_instead_of_adding_a_second_row(self):
        EmbeddingIndex.record(current_provenance())
        EmbeddingIndex.record({**current_provenance(), "model_name": "other"})

        assert EmbeddingIndex.objects.count() == 1
        assert EmbeddingIndex.stored()["model_name"] == "other"

    def test_saving_a_new_instance_still_lands_on_the_single_row(self):
        EmbeddingIndex.record(current_provenance())

        EmbeddingIndex(model_name="m", dimensions=1, text_version=1, text_config={}).save()

        assert list(EmbeddingIndex.objects.values_list("pk", flat=True)) == [EmbeddingIndex.SINGLETON_PK]


@pytest.mark.django_db
class TestGenerateEmbeddings:
    def test_fresh_catalog_is_embedded_and_recorded(self):
        _movie(1)
        _movie(2)

        _generate()

        assert Movie.objects.filter(embedding__isnull=False).count() == 2
        assert EmbeddingIndex.stored() == current_provenance()

    def test_embedder_is_loaded_at_the_pinned_revision(self):
        _movie(1)

        encoder_cls, _, _ = _generate()

        encoder_cls.assert_called_once_with(
            settings.EMBEDDING_MODEL, revision=settings.EMBEDDING_REVISION
        )

    def test_ok_record_embeds_only_the_null_rows(self):
        _record()
        done = _movie(1, embedding=_vec(0.5))
        _movie(2)

        _, encoder, _ = _generate()

        assert len(encoder.encode.call_args.args[0]) == 1
        done.refresh_from_db()
        assert list(done.embedding) == pytest.approx(_vec(0.5))

    def test_mismatch_refuses_without_loading_the_model_or_touching_rows(self):
        _record(model_name="old/model")
        _movie(1, embedding=_vec(0.5))
        _movie(2)

        with patch(f"{GENERATE}.SentenceTransformer") as encoder_cls:
            with pytest.raises(CommandError) as exc:
                call_command("generate_embeddings", stdout=StringIO())

        assert "old/model" in str(exc.value)
        assert "--all" in str(exc.value)
        encoder_cls.assert_not_called()
        assert Movie.objects.filter(embedding__isnull=True).count() == 1
        assert EmbeddingIndex.stored()["model_name"] == "old/model"

    def test_all_re_embeds_every_movie_and_refreshes_the_record(self):
        _record(model_name="old/model")
        _movie(1, embedding=_vec(0.5))
        _movie(2)

        _, encoder, _ = _generate("--all")

        assert len(encoder.encode.call_args.args[0]) == 2
        assert list(Movie.objects.get(tmdb_id=1).embedding) == pytest.approx(_vec(1.0))
        assert EmbeddingIndex.stored() == current_provenance()

    def test_legacy_index_without_a_record_is_refused_with_adopt_hint(self):
        _movie(1, embedding=_vec(0.5))
        _movie(2)

        with patch(f"{GENERATE}.SentenceTransformer") as encoder_cls:
            with pytest.raises(CommandError, match="--adopt"):
                call_command("generate_embeddings", stdout=StringIO())

        encoder_cls.assert_not_called()
        assert EmbeddingIndex.stored() is None

    def test_adopt_records_without_re_embedding(self):
        existing = _movie(1, embedding=_vec(0.5))
        pending = _movie(2)

        encoder_cls, _, _ = _generate("--adopt")

        encoder_cls.assert_not_called()
        assert EmbeddingIndex.stored() == current_provenance()
        existing.refresh_from_db()
        pending.refresh_from_db()
        assert list(existing.embedding) == pytest.approx(_vec(0.5))
        assert pending.embedding is None

    def test_empty_catalog_writes_no_record(self):
        _generate()

        assert EmbeddingIndex.stored() is None


@pytest.mark.django_db
class TestCheckEmbeddingIndex:
    def _run(self):
        out = StringIO()
        call_command("check_embedding_index", stdout=out)
        return out.getvalue()

    def test_ok(self):
        _record()
        _movie(1, embedding=_vec(0.5))

        assert "matches" in self._run()

    def test_no_record_and_no_embeddings_says_generate_has_not_run(self):
        assert "has not run" in self._run()

    def test_no_record_with_embeddings_warns_and_exits_zero(self):
        _movie(1, embedding=_vec(0.5))

        output = self._run()

        assert "WARNING" in output
        assert "--adopt" in output

    def test_mismatch_exits_non_zero_with_differences_and_fix(self):
        _record(dimensions=768)
        _movie(1, embedding=_vec(0.5))

        with pytest.raises(CommandError) as exc:
            self._run()

        assert "dimensions" in str(exc.value)
        assert "generate_embeddings --all" in str(exc.value)

    def test_index_recorded_before_revisions_were_pinned_warns_and_exits_zero(self):
        _record(revision=None)
        _movie(1, embedding=_vec(0.5))

        output = self._run()

        assert "WARNING" in output
        assert "revision" in output

    def test_different_pinned_revision_exits_non_zero(self):
        _record(revision="0" * 40)
        _movie(1, embedding=_vec(0.5))

        with pytest.raises(CommandError) as exc:
            self._run()

        assert "revision" in str(exc.value)
        assert "generate_embeddings --all" in str(exc.value)
