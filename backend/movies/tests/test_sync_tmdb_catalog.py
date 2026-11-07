"""Offline command and startup contract; no live HTTP or database required."""
from datetime import date, datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from django.conf import settings
from django.core.management import call_command
from django.test import override_settings
import pytest
import yaml


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    import socket
    def blocked(*args, **kwargs):
        raise AssertionError('External networking is forbidden in offline tests')
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setattr(socket, 'create_connection', blocked)


@override_settings(TMDB_API_KEY='')
def test_missing_key_skips_without_http():
    out = StringIO()
    with patch('httpx.Client') as client:
        call_command('sync_tmdb_catalog', stdout=out)
    client.assert_not_called()
    assert 'TMDB_API_KEY' in out.getvalue()
    assert 'next startup' in out.getvalue()


@override_settings(TMDB_API_KEY='secret-do-not-log')
def test_success_uses_utc_day_and_settings():
    out = StringIO()
    from movies.catalog_sync import SyncResult
    # Moscow is already on the next day; the command must request the UTC day.
    instant = datetime(2026, 11, 7, 0, 30, tzinfo=timezone(timedelta(hours=3)))
    with patch('movies.catalog_sync.sync_catalog', return_value=SyncResult(9, 2)) as sync, patch('httpx.Client') as http, patch('movies.tmdb_client.TMDBClient') as factory, patch('movies.management.commands.sync_tmdb_catalog.datetime') as clock:
        clock.now.side_effect = lambda tz: instant.astimezone(tz)
        call_command('sync_tmdb_catalog', stdout=out)
    factory.assert_called_once_with(http.return_value.__enter__.return_value, settings.TMDB_PARAMS, api_key='secret-do-not-log')
    clock.now.assert_called_once_with(timezone.utc)
    assert factory.call_args.kwargs == {'api_key': 'secret-do-not-log'}
    assert sync.call_args.args == (factory.return_value,)
    assert sync.call_args.kwargs['today'] == date(2026, 11, 6)
    assert 'checked=9 added=2' in out.getvalue()


@pytest.mark.parametrize('failure', [RuntimeError('url?api_key=secret-do-not-log'), ImportError('secret-do-not-log')])
@override_settings(TMDB_API_KEY='secret-do-not-log')
def test_unexpected_failure_is_safe_and_nonblocking(failure):
    out = StringIO()
    with patch('httpx.Client', side_effect=failure):
        call_command('sync_tmdb_catalog', stdout=out)
    assert 'skipped' in out.getvalue()
    assert 'next startup' in out.getvalue()
    assert 'secret-do-not-log' not in out.getvalue()


def test_common_startup_keeps_blocking_checks_and_guards_sync():
    root = Path(__file__).resolve().parents[3]
    config = yaml.safe_load((root / 'docker-compose.yml').read_text())
    command = config['services']['backend']['command']
    steps = ['migrate --noinput', 'import_catalog --skip-existing', 'check_embedding_index', 'sync_tmdb_catalog', 'uvicorn']
    assert [command.index(step) for step in steps] == sorted(command.index(step) for step in steps)
    assert 'check_embedding_index &&' in command
    assert '(python manage.py sync_tmdb_catalog ||' in command
    assert 'TMDB_API_KEY' in config['services']['backend']['environment']
    for name in ['docker-compose.override.yml', 'docker-compose.openrouter.yml']:
        override = yaml.safe_load((root / name).read_text())
        assert 'command' not in override.get('services', {}).get('backend', {})


@pytest.mark.parametrize('failed_step,server_runs', [
    ('migrate', False), ('import_catalog', False), ('check_embedding_index', False),
    ('sync_tmdb_catalog', True), ('', True),
])
def test_startup_shell_behavior_with_stubbed_commands(failed_step, server_runs):
    # Execute only shell control flow, replacing every application command.
    import shlex
    import subprocess
    root = Path(__file__).resolve().parents[3]
    command = yaml.safe_load((root / 'docker-compose.yml').read_text())['services']['backend']['command']
    body = shlex.split(command)[2]
    stubs = ('python() { [ "$2" != "' + failed_step + '" ]; }; '
             'uvicorn() { echo SERVER_STUB; }; ')
    result = subprocess.run(['sh', '-c', stubs + body], capture_output=True, text=True)
    assert ('SERVER_STUB' in result.stdout) is server_runs
    assert (result.returncode == 0) is server_runs


@override_settings(TMDB_API_KEY='secret-do-not-log')
def test_partial_generation_refusal_is_nonblocking_and_safe():
    from core.embedding_provenance import ProvenanceError
    out = StringIO()
    with patch('movies.catalog_sync.sync_catalog', side_effect=ProvenanceError('secret-do-not-log')), patch('httpx.Client'), patch('movies.tmdb_client.TMDBClient'):
        call_command('sync_tmdb_catalog', stdout=out)
    assert 'index readiness' in out.getvalue()
    assert 'secret-do-not-log' not in out.getvalue()
