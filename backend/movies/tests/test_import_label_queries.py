"""Offline command contract; no database, models or provider calls."""
import csv
import json
from io import StringIO
from unittest.mock import patch

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

MODULE = 'movies.management.commands.import_label_queries'

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError('external networking forbidden')
    monkeypatch.setattr('socket.socket.connect', blocked)
    monkeypatch.setattr('socket.create_connection', blocked)


def inputs(tmp_path):
    source = tmp_path / 'source.csv'
    fields = ['conv_id', 'turn_id', 'turn_order', 'is_seeker', 'processed', 'raw']
    with source.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(dict(zip(fields, ['t3_demo_0/1','t3_demo',0,True,repr(['USER','request']),repr(['USER','Which movies should I watch this evening?'])])))
        writer.writerow(dict(zip(fields, ['t3_demo_0/1','t1_reply',1,False,repr(['SYSTEM','tt0000001']),repr(['SYSTEM','film'])])))
    catalog = tmp_path / 'catalog.csv'
    catalog.write_text('id,imdb_id\n1,tt0000001\n')
    return source, catalog


def run(source, catalog, out):
    with patch(f'{MODULE}._passes_quality_filter', return_value=True), patch(f'{MODULE}.settings.EVAL_LABEL_QUERIES', dict(count=1,test_share=0,seed=7,min_chars=1,max_chars=600,min_linked_recs=1)):
        call_command('import_label_queries', source=str(source), catalog=str(catalog), out=str(out), stdout=StringIO())


def test_explicit_output_and_existing_splits(tmp_path):
    source, catalog = inputs(tmp_path)
    out = tmp_path / 'review' / 'candidate.json'
    out.parent.mkdir()
    out.write_text(json.dumps([dict(id='t3_demo', split='test')]))
    run(source,catalog,out)
    assert json.loads(out.read_text()) == [dict(id='t3_demo',query='Which movies should I watch this evening?',split='test',community_recs=[1])]


@pytest.mark.parametrize('content', ['wrong,headers\na,b\n', 'conv_id,turn_id,turn_order,is_seeker,processed,raw\na,b\n', '\xff'])
def test_malformed_source_actionable(tmp_path, content):
    source,catalog = inputs(tmp_path)
    source.write_bytes(content.encode('latin1'))
    with pytest.raises(CommandError, match='source'):
        run(source,catalog,tmp_path/'out.json')


def test_output_permission_error_actionable(tmp_path):
    source,catalog = inputs(tmp_path)
    with patch(f'{MODULE}.Path.mkdir', side_effect=PermissionError('read-only mount')):
        with pytest.raises(CommandError, match='--out.*writable'):
            run(source,catalog,tmp_path/'out.json')


def test_default_output_uses_writable_review_location():
    from movies.management.commands.import_label_queries import default_output_path
    assert default_output_path().parts[-2:] == ('eval_runs','label_queries.candidate.json')


def test_host_and_container_default_roots(tmp_path):
    from movies.management.commands.import_label_queries import default_output_path
    host_backend = tmp_path / 'repository' / 'backend'
    host_backend.mkdir(parents=True)
    with patch(f'{MODULE}.settings.BASE_DIR', host_backend):
        assert default_output_path() == host_backend.parent / 'eval_runs' / 'label_queries.candidate.json'
    container = tmp_path / 'app'
    (container / 'data').mkdir(parents=True)
    with patch(f'{MODULE}.settings.BASE_DIR', container):
        assert default_output_path() == container / 'eval_runs' / 'label_queries.candidate.json'


def test_invalid_existing_split_file_not_overwritten(tmp_path):
    source, catalog = inputs(tmp_path)
    out = tmp_path / 'out.json'
    out.write_text('[{"id":"t3_demo","split":"invalid"}]')
    before = out.read_bytes()
    with pytest.raises(CommandError, match='existing splits'):
        run(source,catalog,out)
    assert out.read_bytes() == before


def test_missing_source_actionable(tmp_path):
    source,catalog = inputs(tmp_path)
    source.unlink()
    with pytest.raises(CommandError, match='--source file not found'):
        run(source,catalog,tmp_path/'out.json')
