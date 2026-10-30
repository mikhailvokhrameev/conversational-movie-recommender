"""Resolve Compose statically with fictitious secrets; never contact an engine."""
import json
from pathlib import Path
import subprocess
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def resolve_config(tmp_path):
    docker = shutil.which('docker')
    if docker is None:
        pytest.skip('Docker CLI with Compose required for static config validation')
    fixture = tmp_path / 'compose.env'
    fixture.write_text('DB_PASSWORD=fictitious-db\nSECRET_KEY=fictitious-django\n'
                       'OPENROUTER_API_KEY=fictitious-openrouter\n')

    def resolve(*files):
        command = [docker, 'compose', '--env-file', str(fixture)]
        for filename in files:
            command.extend(['-f', filename])
        command.extend(['config', '--format', 'json'])
        result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True,
                                text=True, env={'COMPOSE_DISABLE_ENV_FILE': '1',
                                                'PATH': '/opt/homebrew/bin:/usr/bin:/bin'})
        return json.loads(result.stdout)
    return resolve


def assert_common(config):
    services = config['services']
    backend = services['backend']
    assert Path(backend['build']['context']).name == 'backend'
    assert backend['environment']['PARAMS_PATH'] == '/app/params.yaml'
    assert backend['depends_on']['db']['condition'] == 'service_healthy'
    command = ' '.join(backend['command'])
    for step in ['migrate --noinput', 'import_catalog --skip-existing', 'check_embedding_index']:
        assert step in command
    assert 'generate_embeddings' not in command
    assert 'preflight' not in command
    assert {'pgdata', 'model_cache'} <= set(config['volumes'])
    assert 'name:' not in (ROOT / 'docker-compose.yml').read_text()
    for key, volume in config['volumes'].items():
        assert volume['name'] == f"{config['name']}_{key}"
    for name, service in services.items():
        if name != 'backend':
            assert 'OPENROUTER_API_KEY' not in service.get('environment', {})


def test_default_local_compose(resolve_config):
    config = resolve_config()
    assert_common(config)
    services = config['services']
    assert set(config['volumes']) == {'pgdata', 'ollama_data', 'model_cache'}
    assert set(services) == {'db', 'backend', 'frontend', 'ollama'}
    assert services['backend']['environment']['INFERENCE_MODE'] == 'local'
    assert 'OPENROUTER_API_KEY' not in services['backend']['environment']
    assert services['backend']['depends_on']['ollama']['condition'] == 'service_healthy'
    for name in ['backend', 'ollama']:
        devices = services[name]['deploy']['resources']['reservations']['devices']
        assert devices == [{'capabilities': ['gpu'], 'count': -1, 'driver': 'nvidia'}]


def test_explicit_openrouter_compose(resolve_config):
    config = resolve_config('docker-compose.yml', 'docker-compose.openrouter.yml')
    assert_common(config)
    services = config['services']
    assert set(services) == {'db', 'backend', 'frontend'}
    assert services['backend']['environment']['INFERENCE_MODE'] == 'openrouter'
    assert services['backend']['environment']['OPENROUTER_API_KEY'] == 'fictitious-openrouter'
    assert 'ollama' not in services['backend']['depends_on']
    assert all('deploy' not in service for service in services.values())
