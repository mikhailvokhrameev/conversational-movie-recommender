import logging

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from core.embedding_provenance import (
    ProvenanceError,
    compare_provenance,
    current_provenance,
    plan_generate,
)
from core.embedding_service import build_movie_embedding_text
from movies.models import EmbeddingIndex, Movie

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = (
        "Generate embeddings for movies using sentence-transformers and store in pgvector, "
        "recording which model and text template built the index"
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--batch-size",
            type=int,
            default=None,
            help="Number of descriptions to embed per batch (default: 64)",
        )
        parser.add_argument(
            "--all",
            action="store_true",
            help="Re-embed every movie, not only those without an embedding. Required "
            "after changing the embedding model or text settings.",
        )
        parser.add_argument(
            "--adopt",
            action="store_true",
            help="Record the current config as the provenance of the existing embeddings "
            "without re-embedding. Only for an index built before provenance tracking, "
            "and only if it was built with the current config.",
        )

    def handle(self, *args, **options):
        batch_size = options["batch_size"] or settings.EMBEDDING_BATCH_SIZE

        current = current_provenance()
        check = compare_provenance(EmbeddingIndex.stored(), current)
        try:
            action = plan_generate(
                check,
                Movie.objects.filter(embedding__isnull=False).exists(),
                rebuild_all=options["all"],
                adopt=options["adopt"],
            )
        except ProvenanceError as exc:
            raise CommandError(str(exc)) from exc

        if action == "adopt":
            EmbeddingIndex.record(current)
            self.stdout.write(self.style.SUCCESS(
                "Recorded the current config as the embedding index provenance. "
                "Vectors were not touched; run without --adopt to embed any movies still missing one."
            ))
            return

        # --all re-embeds rows that already have a vector, so it must not filter
        # on embedding IS NULL.
        queryset = Movie.objects.all() if action == "embed_all" else Movie.objects.filter(embedding__isnull=True)
        movies = list(queryset.values_list("id", "serial_name", "genres", "description", "keywords"))

        if not movies:
            self.stdout.write(self.style.SUCCESS("All movies already have embeddings"))
            return

        self.stdout.write(f"Loading embedding model: {settings.EMBEDDING_MODEL}...")
        model = SentenceTransformer(settings.EMBEDDING_MODEL, revision=settings.EMBEDDING_REVISION)
        self.stdout.write(f"Generating embeddings for {len(movies)} movies...")

        updated_ids = []
        updated_vectors = []

        with tqdm(total=len(movies), desc="Embedding", unit="movie") as pbar:
            for i in range(0, len(movies), batch_size):
                batch = movies[i : i + batch_size]

                texts = [
                    build_movie_embedding_text(name, genres, description, keywords)
                    for _, name, genres, description, keywords in batch
                ]

                embeddings = model.encode(texts, convert_to_numpy=True)

                for (movie_id, _, _, _, _), embedding in zip(batch, embeddings):
                    updated_ids.append(movie_id)
                    updated_vectors.append(embedding.tolist())

                pbar.update(len(batch))

        self.stdout.write("Storing embeddings in database...")
        movie_objects = Movie.objects.filter(id__in=updated_ids)
        movie_map = {m.id: m for m in movie_objects}

        to_update = []
        for movie_id, vector in zip(updated_ids, updated_vectors):
            movie = movie_map[movie_id]
            movie.embedding = vector
            to_update.append(movie)

        # One transaction: a failure cannot leave some rows on new vectors and
        # the rest on old ones, nor a record that claims more than was stored.
        with transaction.atomic():
            Movie.objects.bulk_update(to_update, ["embedding"], batch_size=500)
            EmbeddingIndex.record(current)

        embedded_count = Movie.objects.filter(embedding__isnull=False).count()
        total_count = Movie.objects.count()
        self.stdout.write(
            self.style.SUCCESS(
                f"Done. {embedded_count}/{total_count} movies have embeddings"
            )
        )
