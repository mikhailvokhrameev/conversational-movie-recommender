"""Offline contracts for host/container catalog wiring and local data hygiene."""
from pathlib import Path
import runpy
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[3]
CATALOG = 'TMDB Movie Dataset v11.csv'


def load_settings(monkeypatch, settings_file):
    monkeypatch.setenv('PARAMS_PATH', str(ROOT / 'params.yaml'))
    monkeypatch.delenv('CATALOG_CSV_PATH', raising=False)
    monkeypatch.setenv('INFERENCE_MODE', 'local')
    namespace = {'__file__': str(settings_file)}
    exec(compile((ROOT / 'backend/recommender/settings.py').read_text(),
                 str(settings_file), 'exec'), namespace)
    return namespace


def test_host_catalog_default(monkeypatch):
    config = load_settings(monkeypatch, ROOT / 'backend/recommender/settings.py')
    assert Path(config['CATALOG_CSV_PATH']) == ROOT / 'data' / CATALOG


def test_container_catalog_reachable_through_read_only_mount(monkeypatch):
    config = load_settings(monkeypatch, Path('/app/recommender/settings.py'))
    catalog = Path(config['CATALOG_CSV_PATH'])
    assert catalog == Path('/app/data') / CATALOG
    compose = yaml.safe_load((ROOT / 'docker-compose.yml').read_text())
    mounts = compose['services']['backend']['volumes']
    assert './data:/app/data:ro' in mounts
    csv_mounts = [mount for mount in mounts if '.csv:' in mount]
    assert csv_mounts == []


def test_explicit_catalog_override(monkeypatch):
    monkeypatch.setenv('PARAMS_PATH', str(ROOT / 'params.yaml'))
    monkeypatch.setenv('CATALOG_CSV_PATH', '/custom/catalog.csv')
    monkeypatch.setenv('INFERENCE_MODE', 'local')
    config = runpy.run_path(str(ROOT / 'backend/recommender/settings.py'))
    assert config['CATALOG_CSV_PATH'] == '/custom/catalog.csv'


def test_local_csvs_ignored_but_catalog_json_not_ignored():
    paths = ['data/TMDB Movie Dataset v11.csv', 'data/Reddit Movie Large Test.csv']
    result = subprocess.run(['git', 'check-ignore', '--no-index', '--stdin'],
                            input='\n'.join(paths) + '\n', cwd=ROOT,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert result.stdout.splitlines() == paths
    tracked = subprocess.run(['git', 'ls-files', 'data/*.json'], cwd=ROOT,
                             capture_output=True, text=True, check=True)
    assert tracked.stdout.splitlines()
    result = subprocess.run(['git', 'check-ignore', '--no-index', '--stdin'],
                            input=tracked.stdout, cwd=ROOT,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert result.stdout == ''
