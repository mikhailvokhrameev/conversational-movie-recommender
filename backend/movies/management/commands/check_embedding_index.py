from django.core.management.base import BaseCommand, CommandError

from core.embedding_provenance import check_verdict, compare_provenance, current_provenance
from movies.models import EmbeddingIndex, Movie


class Command(BaseCommand):
    help = (
        "Check that the vectors in Movie.embedding were built with the embedding model and "
        "text settings now in params.yaml. Exits non-zero on a mismatch so a start-up chain "
        "stops instead of serving incomparable vectors."
    )

    def handle(self, *args, **options):
        check = compare_provenance(EmbeddingIndex.stored(), current_provenance())
        verdict = check_verdict(check, Movie.objects.filter(embedding__isnull=False).exists())

        if not verdict.healthy:
            raise CommandError(verdict.message)
        if verdict.level == "warning":
            self.stdout.write(self.style.WARNING(f"WARNING: {verdict.message}"))
        else:
            self.stdout.write(verdict.message)
