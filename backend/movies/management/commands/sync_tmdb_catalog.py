"""Best-effort startup update; failures leave the existing catalog available."""
from datetime import datetime, timezone

from django.conf import settings
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Add eligible TMDB films from 2024 onward; failures do not block startup.'

    def handle(self, *args, **options):
        if not settings.TMDB_API_KEY:
            self._skip('TMDB_API_KEY is missing')
            return
        try:
            # Imports are inside the guard: optional preparation failures must
            # not prevent serving the existing catalog. Never print exceptions,
            # URLs or response bodies: they may contain provider credentials.
            import httpx
            from movies.catalog_sync import sync_catalog
            from movies.tmdb_client import TMDBClient

            with httpx.Client() as http:
                client = TMDBClient(http, settings.TMDB_PARAMS, api_key=settings.TMDB_API_KEY)
                result = sync_catalog(client, today=datetime.now(timezone.utc).date())
            self.stdout.write(f'TMDB sync: checked={result.checked} added={result.added}.')
        except Exception:
            self._skip('TMDB retrieval, index readiness or embedding preparation failed')

    def _skip(self, reason):
        self.stdout.write(self.style.WARNING(
            f'TMDB sync skipped: {reason}. Existing catalog remains available. '
            'Check TMDB configuration and embedding-index readiness; retry with '
            '`python manage.py sync_tmdb_catalog` or at the next startup.'
        ))
