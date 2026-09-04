import logging

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand

from movies.models import Movie
from movies.search_index import refresh_search_vectors

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Import movie catalog from catalog_okko.parquet into the database"

    def add_arguments(self, parser):
        parser.add_argument(
            "--skip-existing",
            action="store_true",
            help="Skip movies that already exist (matched by URL)",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=None,
            help="Movies per insert batch (default: catalog.import_batch_size)",
        )

    def handle(self, *args, **options):
        skip_existing = options["skip_existing"]
        batch_size = options["batch_size"] or settings.IMPORT_BATCH_SIZE
        parquet_path = settings.CATALOG_PARQUET_PATH

        self.stdout.write(f"Reading catalog from {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        self.stdout.write(f"Loaded {len(df)} items from parquet")

        if skip_existing:
            existing_urls = set(Movie.objects.values_list("url", flat=True))
            df = df[~df["url"].isin(existing_urls)]
            self.stdout.write(f"{len(df)} new items to import (skipped existing)")

        if df.empty:
            self.stdout.write(self.style.SUCCESS("Nothing to import"))
            return

        movies = []
        for _, row in df.iterrows():
            genres = row["genres"]
            if not isinstance(genres, list):
                genres = list(genres) if genres is not None else []

            country = row["country"]
            if not isinstance(country, list):
                country = list(country) if country is not None else []

            actors = row["actors"]
            if not isinstance(actors, list):
                actors = list(actors) if actors is not None else []

            release_date = row.get("release_date")
            if pd.isna(release_date):
                release_date = None

            age_rating = row.get("age_rating")
            if pd.isna(age_rating):
                age_rating = None

            movies.append(
                Movie(
                    serial_name=row["serial_name"] or "",
                    genres=genres,
                    content_type=row["content_type"] or "",
                    country=country,
                    actors=actors,
                    director=row["director"] or "",
                    age_rating=age_rating,
                    studio_name=row["studio_name"] or "",
                    release_date=release_date,
                    description=row["description"] or "",
                    url=row["url"],
                )
            )

        created = 0
        for i in range(0, len(movies), batch_size):
            batch = movies[i : i + batch_size]
            Movie.objects.bulk_create(batch, ignore_conflicts=True)
            created += len(batch)
            self.stdout.write(f"  Imported {created}/{len(movies)}...")

        # bulk_create bypasses any per-row vector computation, so the lexical
        # index is rebuilt here in one UPDATE.
        self.stdout.write("Building full-text search vectors...")
        indexed = refresh_search_vectors(only_missing=skip_existing)
        self.stdout.write(f"  Indexed {indexed} movies")

        total = Movie.objects.count()
        self.stdout.write(
            self.style.SUCCESS(f"Done. Total movies in database: {total}")
        )
