import csv
import json
import logging
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from core.label_queries import imdb_to_tmdb, read_turns, select_queries
from movies.management.commands.import_catalog import _passes_quality_filter

logger = logging.getLogger(__name__)

def project_root():
    base = Path(settings.BASE_DIR)
    return base if (base / "data").is_dir() else base.parent


def default_output_path():
    return project_root() / "eval_runs" / "label_queries.candidate.json"


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
            default=None,
            help="Candidate output (default: project eval_runs/label_queries.candidate.json; writable container mount)",
        )

        parser.add_argument("--existing", help="Existing query JSON whose split identities must be retained (default: existing output, then project data/label_queries.json)")

    def handle(self, *args, **options):
        source = Path(options["source"])
        catalog = Path(options["catalog"] or settings.CATALOG_CSV_PATH)
        out = Path(options["out"]) if options["out"] else default_output_path()
        for label, path in (("source", source), ("catalog", catalog)):
            if not path.is_file():
                raise CommandError(f"--{label} file not found: {path}")

        params = settings.EVAL_LABEL_QUERIES
        min_vote_count = settings.MIN_VOTE_COUNT
        # Only films the import keeps can be recommended, so only those count.
        try:
            imdb_map = imdb_to_tmdb(
                catalog, row_filter=lambda row: _passes_quality_filter(row, min_vote_count)
            )
        except (OSError, UnicodeError, csv.Error, ValueError) as exc:
            raise CommandError(f"Cannot read --catalog {catalog}: {exc}") from exc
        self.stdout.write(f"{len(imdb_map)} catalog films have an IMDb id")
        try:
            records = read_turns(source)
        except (OSError, UnicodeError, csv.Error, ValueError) as exc:
            raise CommandError(f"Cannot read --source {source}: {exc}") from exc
        existing = Path(options["existing"]) if options.get("existing") else (
            out if out.exists() else project_root() / "data" / "label_queries.json"
        )
        splits = {}
        if existing.exists() or options.get("existing"):
            try:
                rows = json.loads(existing.read_text(encoding="utf-8"))
                if not isinstance(rows, list):
                    raise ValueError("expected a JSON list")
                for row in rows:
                    if row["id"] in splits or row["split"] not in {"dev", "test"}:
                        raise ValueError("duplicate ID or invalid split")
                    splits[row["id"]] = row["split"]
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise CommandError(f"Cannot read existing splits {existing}: {exc}") from exc
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
            existing_splits=splits,
        )
        if len(selected) < params["count"]:
            self.stderr.write(
                f"Only {len(selected)} usable requests found (wanted {params['count']})."
            )

        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as handle:
                json.dump(selected, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
        except OSError as exc:
            raise CommandError(f"Cannot write --out {out}: {exc}; choose a writable path (container: /app/eval_runs/)") from exc
        n_test = sum(r["split"] == "test" for r in selected)
        self.stdout.write(
            f"Wrote {len(selected)} queries ({len(selected) - n_test} dev, {n_test} test) to {out}. "
            "Delete non-requests and non-English ones by hand before committing."
        )
