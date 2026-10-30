from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from core.embedding_provenance import (
    ProvenanceError, check_verdict, compare_provenance, current_provenance,
    validate_compatibility_vectors,
)
from core.embedding_service import build_movie_embedding_text, encode_texts
from core.openrouter_client import InferenceError
from movies.models import EmbeddingIndex, Movie


class Command(BaseCommand):
    help = (
        "Check embedding provenance and sample compatibility after a backend change. "
        "Exits non-zero on a mismatch so start-up cannot serve incomparable vectors."
    )

    def handle(self, *args, **options):
        check = compare_provenance(EmbeddingIndex.stored(), current_provenance())
        embedded = Movie.objects.filter(embedding__isnull=False)
        has_embeddings = embedded.exists()
        if check.status == "backend_changed" and has_embeddings:
            config = settings.EFFECTIVE_PARAMS["embedding"]["compatibility"]
            rows = list(embedded.order_by("tmdb_id").values_list(
                "tmdb_id", "serial_name", "genres", "description", "keywords", "embedding",
            )[:config["sample_size"]])
            texts = [build_movie_embedding_text(*row[1:5]) for row in rows]
            try:
                validate_compatibility_vectors(
                    [row[5] for row in rows], encode_texts(texts), config["min_cosine"],
                )
            except (ProvenanceError, InferenceError) as exc:
                raise CommandError(
                    "Embedding compatibility check failed. Use the original inference mode, "
                    "or explicitly run `manage.py generate_embeddings --all` to rebuild the index."
                ) from exc
            self.stdout.write(
                f"Embedding backend compatibility confirmed for {len(rows)} sampled movies "
                f"(cosine >= {config['min_cosine']}). Origin unchanged; this permits reading "
                "only, does not prove all vectors identical, and partial generation still requires "
                "the original origin or explicit --all."
            )
            for note in check.notes:
                self.stdout.write(self.style.WARNING(f"WARNING: {note}"))
            return

        verdict = check_verdict(check, has_embeddings)
        if not verdict.healthy:
            raise CommandError(verdict.message)
        if verdict.level == "warning":
            self.stdout.write(self.style.WARNING(f"WARNING: {verdict.message}"))
        else:
            self.stdout.write(verdict.message)
