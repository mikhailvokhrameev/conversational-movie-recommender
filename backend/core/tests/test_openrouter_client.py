from unittest.mock import Mock, AsyncMock
import httpx
import pytest
from django.test import override_settings
from core import openrouter_client as c

@pytest.fixture(autouse=True)
def config(monkeypatch):
    with override_settings(OPENROUTER_API_KEY='secret', EFFECTIVE_PARAMS={'openrouter': {'base_url': 'https://example.invalid/v1', 'retry': {'max_attempts': 3, 'initial_delay': .5, 'max_delay': 5}}}):
        monkeypatch.setattr(c.time, 'sleep', Mock())
        monkeypatch.setattr(c.asyncio, 'sleep', AsyncMock())
        yield

def reply(status, **kwargs):
    return httpx.Response(status, **kwargs)

@pytest.mark.parametrize('status', [400,401,402,403])
def test_permanent(monkeypatch, caplog, status):
    post=Mock(return_value=reply(status, json={'error':'secret private-prompt'}))
    monkeypatch.setattr(httpx.Client,'post',post)
    with pytest.raises(c.InferenceError) as error:
        c.post_json('chat/completions', {'prompt':'private-prompt'}, timeout=7)
    assert post.call_count == 1
    assert error.value.status_code == status
    assert not error.value.retryable
    assert error.value.__context__ is None
    assert 'secret' not in str(error.value)+caplog.text
    assert 'private-prompt' not in str(error.value)+caplog.text

def test_authorization_retry(monkeypatch):
    post=Mock(side_effect=[reply(429, headers={'Retry-After':'999'}),reply(200,json={'ok':True})])
    monkeypatch.setattr(httpx.Client,'post',post)
    assert c.post_json('/embeddings', {'input':'text'}, timeout=7)=={'ok':True}
    assert post.call_args.args == ('https://example.invalid/v1/embeddings',)
    assert post.call_args.kwargs == {'json':{'input':'text'},'headers':{'Authorization':'Bearer secret'},'timeout':7}
    c.time.sleep.assert_called_once_with(5)

@pytest.mark.parametrize('failure',[reply(503),httpx.ReadTimeout('secret')])
def test_exhaustion(monkeypatch,failure):
    post=Mock(side_effect=[failure]*3)
    monkeypatch.setattr(httpx.Client,'post',post)
    with pytest.raises(c.InferenceError) as error:
        c.post_json('embeddings',{},timeout=2)
    assert post.call_count==3
    assert error.value.retryable
    assert error.value.__context__ is None
    assert 'secret' not in str(error.value)
    assert [call.args[0] for call in c.time.sleep.call_args_list]==[.5,1]

@pytest.mark.parametrize('content',[b'[]',b'null',b'broken'])
def test_invalid_success(monkeypatch,content):
    post=Mock(return_value=reply(200,content=content))
    monkeypatch.setattr(httpx.Client,'post',post)
    with pytest.raises(c.InferenceError): c.post_json('embeddings',{},timeout=1)
    assert post.call_count==1

def test_missing_key(monkeypatch):
    post=Mock()
    monkeypatch.setattr(httpx.Client,'post',post)
    with override_settings(OPENROUTER_API_KEY=''):
        with pytest.raises(c.InferenceError): c.post_json('embeddings',{},timeout=1)
    post.assert_not_called()

@pytest.mark.asyncio
async def test_async(monkeypatch):
    post=AsyncMock(side_effect=[httpx.ConnectError('secret'),reply(200,json={'ok':True})])
    monkeypatch.setattr(httpx.AsyncClient,'post',post)
    assert await c.apost_json('embeddings',{},timeout=2)=={'ok':True}
    assert post.call_count==2
    assert post.call_args.kwargs['headers']=={'Authorization':'Bearer secret'}
    c.asyncio.sleep.assert_awaited_once_with(.5)

@pytest.mark.asyncio
@pytest.mark.parametrize('failure', [reply(401), reply(402), reply(503), httpx.ReadTimeout('secret'), reply(200, content=b'[]')])
async def test_async_failures(monkeypatch, failure):
    post = AsyncMock(side_effect=[failure] * 3)
    monkeypatch.setattr(httpx.AsyncClient, 'post', post)
    with pytest.raises(c.InferenceError) as error:
        await c.apost_json('embeddings', {}, timeout=2)
    retryable = isinstance(failure, httpx.RequestError) or failure.status_code == 503
    assert error.value.retryable == retryable
    assert post.call_count == (3 if retryable else 1)
    assert error.value.__context__ is None
    assert 'secret' not in str(error.value)

@pytest.mark.parametrize('endpoint', ['https://evil.invalid', '//evil.invalid?key=secret', '../chat', 'chat#secret'])
def test_reject_external_endpoints(monkeypatch, endpoint):
    post = Mock()
    monkeypatch.setattr(httpx.Client, 'post', post)
    with pytest.raises(c.InferenceError):
        c.post_json(endpoint, {}, timeout=1)
    post.assert_not_called()

@pytest.mark.parametrize('value,expected', [('garbage', .5), ('-20', 0), ('nan', .5), ('inf', 5), ('Thu, 01 Jan 1970 00:00:00 GMT', 0)])
def test_retry_after_values(value, expected):
    policy = {'initial_delay': .5, 'max_delay': 5}
    assert c._delay(policy, 0, reply(429, headers={'Retry-After': value})) == expected
