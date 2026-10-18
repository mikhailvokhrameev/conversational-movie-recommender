"""Print how each intent.resolution field resolves the calibration probes.

Usage:
  python manage.py entity_resolution_report                  # data/entity_probes.json
  python manage.py entity_resolution_report --probes p.json

Dev-only. For every probe in data/entity_probes.json it prints the resolved
catalog value, the layer that matched (exact / synonym / embedding / dropped)
and, for fields with the embedding fallback, the embedding's top-2 catalog
values with their cosine scores and margin -- even when an earlier layer
matched, so every probe shows what the embedding alone would pick.
match_threshold and match_margin in params.yaml are set by hand from this
output; the command never writes params.yaml. It needs the DB (country and
language catalogs) and the real embedder, so run it on the GPU machine.
"""

import json
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from core.ollama_client import RESOLVERS


def default_probes_path() -> Path:
    """data/entity_probes.json, found the way default_golden_set_path finds the
    golden set: the container location first, then the repo root."""
    candidates = [
        Path(settings.BASE_DIR) / "data" / "entity_probes.json",
        Path(settings.BASE_DIR).parent / "data" / "entity_probes.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[-1]


def _load_probes(path: Path) -> dict[str, list[str]]:
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise CommandError(f"Cannot read probes file {path}: {e}")
    if not isinstance(raw, dict):
        raise CommandError(f"{path}: expected an object mapping resolver names to probe lists.")
    probes = {k: v for k, v in raw.items() if not k.startswith("_")}
    unknown = set(probes) - set(RESOLVERS)
    if unknown:
        raise CommandError(
            f"{path}: unknown fields {', '.join(sorted(unknown))}; expected {', '.join(RESOLVERS)}."
        )
    for field, values in probes.items():
        if not isinstance(values, list) or not all(isinstance(v, str) and v.strip() for v in values):
            raise CommandError(f"{path}: {field} must be a list of non-blank strings.")
    return probes


def _fmt_top(top) -> str:
    scores = ", ".join(f"'{name}' {score:.3f}" for name, score in top)
    if len(top) > 1:
        scores += f" (margin {top[0][1] - top[1][1]:.3f})"
    return scores


class Command(BaseCommand):
    help = "Print how intent.resolution resolves the calibration probes (dev-only)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--probes",
            type=str,
            default=None,
            help="Path to the probes JSON (default: data/entity_probes.json)",
        )

    def handle(self, *args, **options):
        path = Path(options["probes"]) if options["probes"] else default_probes_path()
        probes = _load_probes(path)
        for field, values in probes.items():
            resolver = RESOLVERS[field]
            if resolver.uses_embeddings:
                header = (
                    f"{field} (match_threshold {resolver.match_threshold}, "
                    f"match_margin {resolver.match_margin})"
                )
                tops = resolver.nearest(values)
            else:
                header = f"{field} (no embedding fallback)"
                tops = [()] * len(values)
            self.stdout.write(header)
            width = max((len(v) for v in values), default=0)
            for result, top in zip(resolver.resolve_each(values), tops):
                value = result.value if result.value is not None else "(dropped)"
                line = f"  {result.raw:<{width}}  -> {value}  [{result.via}]"
                if top:
                    line += f"  top-2: {_fmt_top(top)}"
                self.stdout.write(line)
            self.stdout.write("")
