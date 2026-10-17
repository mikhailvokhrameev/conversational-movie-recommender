import csv
import logging
from datetime import date

from django.conf import settings
from django.core.management.base import BaseCommand

from movies.models import Movie
from movies.search_index import refresh_search_vectors

logger = logging.getLogger(__name__)

# Raised well above Python's 128KB default: TMDB overview/keywords fields can
# be long enough to trip it on a handful of rows.
csv.field_size_limit(10_000_000)


def _split_list(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(", ") if item.strip()]


def _parse_date(value: str) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _parse_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _passes_quality_filter(row: dict, min_vote_count: int) -> bool:
    return (
        row.get("status") == "Released"
        and row.get("adult") != "True"
        and bool((row.get("overview") or "").strip())
        and bool((row.get("poster_path") or "").strip())
        and _parse_int(row.get("vote_count")) >= min_vote_count
    )


def _row_to_movie(row: dict) -> Movie:
    return Movie(
        tmdb_id=_parse_int(row["id"]),
        serial_name=row.get("title") or "",
        original_title=row.get("original_title") or "",
        genres=_split_list(row.get("genres") or ""),
        country=_split_list(row.get("production_countries") or ""),
        original_language=row.get("original_language") or "",
        keywords=_split_list(row.get("keywords") or ""),
        release_date=_parse_date(row.get("release_date") or ""),
        description=row.get("overview") or "",
        runtime=_parse_int(row.get("runtime"), default=None) if row.get("runtime") else None,
        popularity=_parse_float(row.get("popularity")),
        vote_average=_parse_float(row.get("vote_average")),
        vote_count=_parse_int(row.get("vote_count")),
        poster_path=row.get("poster_path") or None,
    )


class Command(BaseCommand):
    help = "Import the TMDB movie catalog CSV into the database"

    def add_arguments(self, parser):
        parser.add_argument(
            "--skip-existing",
            action="store_true",
            help="Skip movies that already exist (matched by tmdb_id)",
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
        csv_path = settings.CATALOG_CSV_PATH
        min_vote_count = settings.MIN_VOTE_COUNT

        existing_ids = (
            set(Movie.objects.values_list("tmdb_id", flat=True)) if skip_existing else set()
        )

        self.stdout.write(f"Reading catalog from {csv_path}...")
        seen = 0
        kept = 0
        batch: list[Movie] = []
        created = 0

        with open(csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                seen += 1
                if not _passes_quality_filter(row, min_vote_count):
                    continue

                tmdb_id = _parse_int(row["id"])
                if skip_existing and tmdb_id in existing_ids:
                    continue

                kept += 1
                batch.append(_row_to_movie(row))

                if len(batch) >= batch_size:
                    Movie.objects.bulk_create(batch, ignore_conflicts=True)
                    created += len(batch)
                    self.stdout.write(f"  Imported {created} ({seen} rows scanned)...")
                    batch = []

        if batch:
            Movie.objects.bulk_create(batch, ignore_conflicts=True)
            created += len(batch)

        self.stdout.write(f"Scanned {seen} rows, kept {kept} passing the quality filter")

        if created == 0:
            self.stdout.write(self.style.SUCCESS("Nothing to import"))
            return

        # bulk_create bypasses any per-row vector computation, so the lexical
        # index is rebuilt here in one UPDATE.
        self.stdout.write("Building full-text search vectors...")
        indexed = refresh_search_vectors(only_missing=skip_existing)
        self.stdout.write(f"  Indexed {indexed} movies")

        total = Movie.objects.count()
        self.stdout.write(
            self.style.SUCCESS(f"Done. Total movies in database: {total}")
        )
