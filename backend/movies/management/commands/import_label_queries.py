import csv
import json
import logging
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from core.label_queries import imdb_to_tmdb, select_queries
from movies.management.commands.import_catalog import _passes_quality_filter

logger = logging.getLogger(__name__)

csv.field_size_limit(10_000_000)


class Command(BaseCommand):
    help = (
        "Build data/label_queries.json from a downloaded Reddit-Movie CSV: movie "
        "requests plus the catalog films the replies recommended, for hand grading"
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--source",
            required=True,
            help="Downloaded Reddit-Movie CSV (e.g. test.csv from ZhankuiHe/reddit_movie_large_v1)",
        )
        parser.add_argument(
            "--catalog",
            default=None,
            help="TMDB catalog CSV (default: the CSV import_catalog reads)",
        )
        parser.add_argument(
            "--out",
            default="data/label_queries.json",
            help="Where to write the query set (default: data/label_queries.json)",
        )

    def handle(self, *args, **options):
        source = Path(options["source"])
        catalog = Path(options["catalog"] or settings.CATALOG_CSV_PATH)
        out = Path(options["out"])
        for label, path in (("source", source), ("catalog", catalog)):
            if not path.is_file():
                raise CommandError(f"--{label} file not found: {path}")

        params = settings.EVAL_LABEL_QUERIES
        min_vote_count = settings.MIN_VOTE_COUNT
        # Only films the import keeps can be recommended, so only those count.
        imdb_map = imdb_to_tmdb(
            catalog, row_filter=lambda row: _passes_quality_filter(row, min_vote_count)
        )
        self.stdout.write(f"{len(imdb_map)} catalog films have an IMDb id")

        with open(source, newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
        self.stdout.write(f"Read {len(records)} turns from {source}")

        selected = select_queries(
            records,
            imdb_map,
            count=params["count"],
            test_share=params["test_share"],
            seed=params["seed"],
            min_chars=params["min_chars"],
            max_chars=params["max_chars"],
            min_linked_recs=params["min_linked_recs"],
        )
        if len(selected) < params["count"]:
            self.stderr.write(
                f"Only {len(selected)} usable requests found (wanted {params['count']})."
            )

        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(selected, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        n_test = sum(r["split"] == "test" for r in selected)
        self.stdout.write(
            f"Wrote {len(selected)} queries ({len(selected) - n_test} dev, {n_test} test) to {out}. "
            "Delete non-requests and non-English ones by hand before committing."
        )
