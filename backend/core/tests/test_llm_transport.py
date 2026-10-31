"""Offline HTTP contracts shared by the two LLM providers."""
import json
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from core.llm_transport import OllamaLLMAdapter, OpenRouterLLMAdapter
from core.openrouter_client import InferenceError


@pytest.fixture(autouse=True)
def config(settings, monkeypatch):
    settings.OPENROUTER_API_KEY = 'test-secret'
    settings.LLM_MODEL = 'test-model'
    settings.LLM_BASE_URL = 'http://ollama.invalid'
    settings.EFFECTIVE_PARAMS = {**settings.EFFECTIVE_PARAMS, 'openrouter': {
        'base_url': 'https://openrouter.invalid/v1',
        'retry': {'max_attempts': 3, 'initial_delay': .01, 'max_delay': .02},
    }}
    monkeypatch.setattr('core.openrouter_client.time.sleep', Mock())
    monkeypatch.setattr('core.openrouter_client.asyncio.sleep', AsyncMock())


def kwargs():
    return dict(messages=[{'role': 'user', 'content': 'Hello'}], schema=None,
                kind='generation', timeout=7, call_type='test')


def sse(*events):
    return ''.join('data: ' + (e if isinstance(e, str) else json.dumps(e)) + '\n\n' for e in events)


def chunk(content):
    return {'choices': [{'delta': {'content': content}}]}


@pytest.mark.parametrize('provider', ['ollama', 'openrouter'])
def test_complete_wire_contract(monkeypatch, provider, settings):
    body = {'message': {'content': 'Hello'}} if provider == 'ollama' else {
        'choices': [{'message': {'content': 'Hello', 'reasoning': 'private'}}]}
    response = httpx.Response(200, json=body, request=httpx.Request('POST', 'http://test.invalid'))
    post = Mock(return_value=response)
    monkeypatch.setattr(httpx, 'post', post)
    monkeypatch.setattr(httpx.Client, 'post', post)
    adapter = OllamaLLMAdapter() if provider == 'ollama' else OpenRouterLLMAdapter()
    assert adapter.complete(**kwargs()) == 'Hello'
    payload = post.call_args.kwargs['json']
    assert payload['model'] == 'test-model'
    if provider == 'ollama':
        assert payload['think'] is False
        assert payload['options']['num_ctx'] == settings.LLM_NUM_CTX
    else:
        assert payload['reasoning'] == {'enabled': False}
        assert payload['provider']['require_parameters'] is True
        assert payload['max_tokens'] == settings.LLM_MAX_OUTPUT_TOKENS['generation']
        assert 'num_ctx' not in payload


def test_strict_schema_does_not_mutate_input(monkeypatch):
    post = Mock(return_value=httpx.Response(200, json={'choices': [{'message': {'content': '{}'}}]}))
    monkeypatch.setattr(httpx.Client, 'post', post)
    schema = {'type': 'object', 'properties': {'nested': {'type': 'object', 'properties': {'x': {'type': 'string'}}}}}
    OpenRouterLLMAdapter().complete(**{**kwargs(), 'schema': schema, 'kind': 'structured'})
    wire = post.call_args.kwargs['json']['response_format']['json_schema']
    assert wire['strict'] is True
    assert wire['schema']['additionalProperties'] is False
    assert wire['schema']['properties']['nested']['additionalProperties'] is False
    assert wire['schema']['required'] == ['nested']
    assert 'additionalProperties' not in schema


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.asyncio
async def test_sse_comments_usage_split_lines_and_reasoning(monkeypatch, asynchronous):
    content = ': keepalive\n\nevent: message\ndata: {"choices":\ndata: [{"delta":{"reasoning":"secret"}}]}\n\n'
    content += sse(chunk('Hi'), {'usage': {'prompt_tokens': 5}, 'choices': []}, chunk(' there'), '[DONE]')
    class Bytes(httpx.SyncByteStream, httpx.AsyncByteStream):
        def __iter__(self):
            for i in range(0, len(content), 3):
                yield content[i:i+3].encode()
        async def __aiter__(self):
            for part in self:
                yield part
    transport = httpx.MockTransport(lambda request: httpx.Response(200, stream=Bytes()))
    original_client, original_async = httpx.Client, httpx.AsyncClient
    monkeypatch.setattr(httpx, 'Client', lambda **kw: original_client(transport=transport, **kw))
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: original_async(transport=transport, **kw))
    adapter = OpenRouterLLMAdapter()
    tokens = [t async for t in adapter.astream(**kwargs())] if asynchronous else list(adapter.stream(**kwargs()))
    assert tokens == ['Hi', ' there']


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('prefix', [
    sse(chunk('first')),
    sse({'choices': [{'delta': {'reasoning': 'private'}}]}),
    sse({'usage': {'prompt_tokens': 5}, 'choices': []}),
    ': keepalive\n\n',
    '',
])
@pytest.mark.parametrize('ending', ['error', 'incomplete'])
@pytest.mark.asyncio
async def test_started_stream_is_never_retried(monkeypatch, asynchronous, prefix, ending):
    calls = []
    def handler(request):
        calls.append(request)
        terminal = sse({'error': {'code': 503, 'message': 'test-secret'}}) if ending == 'error' else ''
        return httpx.Response(200, text=prefix + terminal)
    transport = httpx.MockTransport(handler)
    original_client, original_async = httpx.Client, httpx.AsyncClient
    monkeypatch.setattr(httpx, 'Client', lambda **kw: original_client(transport=transport, **kw))
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: original_async(transport=transport, **kw))
    adapter = OpenRouterLLMAdapter()
    with pytest.raises(InferenceError) as error:
        if asynchronous:
            [token async for token in adapter.astream(**kwargs())]
        else:
            list(adapter.stream(**kwargs()))
    assert len(calls) == 1
    assert 'test-secret' not in str(error.value)
    assert error.value.__context__ is None


@pytest.mark.parametrize('provider', ['ollama', 'openrouter'])
@pytest.mark.asyncio
async def test_common_intent_repair_history_and_selected_movie(monkeypatch, provider):
    from core import llm_client as client
    from core.inference_backends import InferenceBackends
    bodies = iter(['{"category": "invalid"}', '{"category":"general_chat","wants_obscure":true}'])
    requests = []
    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if payload['stream']:
            text = '<think>private</think>Hello'
            wire = json.dumps({'message': {'content': text}}) + '\n' if provider == 'ollama' else sse(chunk(text), '[DONE]')
            return httpx.Response(200, text=wire)
        text = next(bodies)
        body = {'message': {'content': text}} if provider == 'ollama' else {'choices': [{'message': {'content': text}}]}
        return httpx.Response(200, json=body)
    transport = httpx.MockTransport(handler)
    original_async = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: original_async(transport=transport, **kw))
    adapter = OllamaLLMAdapter() if provider == 'ollama' else OpenRouterLLMAdapter()
    monkeypatch.setattr(client, 'get_inference_backends', lambda: InferenceBackends(adapter, None, None))
    monkeypatch.setattr(client, 'catalog_year_range', lambda: None)
    monkeypatch.setattr(client, '_resolve_fields', lambda fields: fields)
    history = [{'role': 'user', 'content': 'I liked Heat'}]
    intent = await client.aclassify_and_parse('this one?', history, selection_note='Selected Heat')
    assert intent.category == 'general_chat'
    assert intent.wants_obscure is True
    assert 'invalid' in requests[1]['messages'][-1]['content']
    assert requests[0]['messages'][1]['content'] == 'I liked Heat'
    assert 'Selected Heat' in requests[1]['messages'][-1]['content']
    tokens = [token async for token in client.astream_conversational('this?', history,
        selected_movie={'serial_name': 'Heat', 'release_date': '1995-01-01', 'genres': ['Crime']}, selected_position=1)]
    assert ''.join(tokens) == 'Hello'
    assert 'Heat' in requests[-1]['messages'][-1]['content']


@pytest.mark.parametrize('provider', ['ollama', 'openrouter'])
@pytest.mark.asyncio
async def test_common_http_error_keeps_provider_and_falls_back(monkeypatch, provider, caplog):
    from core import llm_client as client
    from core.inference_backends import InferenceBackends
    adapter = OllamaLLMAdapter() if provider == 'ollama' else OpenRouterLLMAdapter()
    monkeypatch.setattr(client, 'get_inference_backends', lambda: InferenceBackends(adapter, None, None))
    monkeypatch.setattr(client, 'catalog_year_range', lambda: None)
    post = AsyncMock(return_value=httpx.Response(401, text='test-secret',
        request=httpx.Request('POST', 'https://provider.invalid/chat')))
    monkeypatch.setattr(httpx.AsyncClient, 'post', post)
    intent = await client.aclassify_and_parse('a comedy')
    assert intent.category == 'new_search'
    assert intent.semantic_query == 'a comedy'
    assert intent.genres == []
    assert post.call_count == 1
    assert adapter.provider == provider
    assert 'using fallback' in caplog.text
    assert 'test-secret' not in caplog.text


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('status,attempts', [(401, 1), (429, 3), (503, 3)])
@pytest.mark.asyncio
async def test_stream_http_errors_are_bounded_and_safe(monkeypatch, asynchronous, status, attempts):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(status, text='test-secret', headers={'Retry-After': '999'})
    transport = httpx.MockTransport(handler)
    original_client, original_async = httpx.Client, httpx.AsyncClient
    monkeypatch.setattr(httpx, 'Client', lambda **kw: original_client(transport=transport, **kw))
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: original_async(transport=transport, **kw))
    adapter = OpenRouterLLMAdapter()
    with pytest.raises(InferenceError) as error:
        if asynchronous:
            [t async for t in adapter.astream(**kwargs())]
        else:
            list(adapter.stream(**kwargs()))
    assert len(calls) == attempts
    assert error.value.status_code == status
    assert 'test-secret' not in str(error.value)
    assert error.value.__context__ is None
