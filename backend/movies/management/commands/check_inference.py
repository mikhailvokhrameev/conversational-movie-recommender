"""Explicit provider probes; never run automatically at application startup."""
from django.core.management.base import BaseCommand
from core.inference_diagnostics import check_inference


class Command(BaseCommand):
    help = 'Check configured inference providers using small real requests (may incur API costs).'

    def handle(self, *args, **options):
        metadata = check_inference()
        self.stdout.write(f"Inference mode: {metadata['mode']}")
        for component, config in metadata['components'].items():
            self.stdout.write(f"{component}: {config['provider']} / {config.get('api_model', config['model'])}")
        self.stdout.write(self.style.SUCCESS('Inference checks passed.'))
