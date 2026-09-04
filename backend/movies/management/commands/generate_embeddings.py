import logging

from django.conf import settings
from django.core.management.base import BaseCommand
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from movies.models import Movie

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Generate embeddings for movies using sentence-transformers and store in pgvector"

    def add_arguments(self, parser):
        parser.add_argument(
            "--batch-size",
            type=int,
            default=None,
            help="Number of descriptions to embed per batch (default: 64)",
        )

    def handle(self, *args, **options):
        batch_size = options["batch_size"] or settings.EMBEDDING_BATCH_SIZE

        movies = list(
            Movie.objects.filter(embedding__isnull=True)
            .values_list("id", "serial_name", "genres", "description")
        )

        if not movies:
            self.stdout.write(self.style.SUCCESS("All movies already have embeddings"))
            return

        self.stdout.write(f"Loading embedding model: {settings.EMBEDDING_MODEL}...")
        model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.stdout.write(f"Generating embeddings for {len(movies)} movies...")

        updated_ids = []
        updated_vectors = []

        with tqdm(total=len(movies), desc="Embedding", unit="movie") as pbar:
            for i in range(0, len(movies), batch_size):
                batch = movies[i : i + batch_size]

                texts = []
                for movie_id, name, genres, description in batch:
                    genre_str = ", ".join(genres) if genres else ""
                    desc_truncated = (description or "")[:500]
                    texts.append(f"{name}. {genre_str}. {desc_truncated}")

                embeddings = model.encode(texts, convert_to_numpy=True)

                for (movie_id, _, _, _), embedding in zip(batch, embeddings):
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

        Movie.objects.bulk_update(to_update, ["embedding"], batch_size=500)

        embedded_count = Movie.objects.filter(embedding__isnull=False).count()
        total_count = Movie.objects.count()
        self.stdout.write(
            self.style.SUCCESS(
                f"Done. {embedded_count}/{total_count} movies have embeddings"
            )
        )
