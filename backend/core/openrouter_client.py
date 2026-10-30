"""Small, non-streaming OpenRouter JSON transport with bounded retries.

Errors deliberately retain neither upstream bodies nor HTTP exception objects.
Streaming adapters must handle their own stream without replaying consumed tokens.
"""
import asyncio
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import time

import httpx
from django.conf import settings


class InferenceError(Exception):
    """Public transport failure containing only safe operational metadata."""

    def __init__(self, operation: str, status_code: int | None = None,
                 retryable: bool = False):
        self.operation = operation
        self.status_code = status_code
        self.retryable = retryable
        super().__init__(f"OpenRouter {operation} failed (status={status_code}, retryable={retryable})")


def _request(endpoint):
    # Never put a caller-supplied endpoint into diagnostics; it could contain secrets.
    key = settings.OPENROUTER_API_KEY
    if not key:
        raise InferenceError("post_json")
    config = settings.EFFECTIVE_PARAMS["openrouter"]
    # Only relative API paths are accepted: credentials must not follow arbitrary URLs.
    if not endpoint or ":" in endpoint or "?" in endpoint or "#" in endpoint or ".." in endpoint:
        raise InferenceError("post_json")
    return (config["base_url"].rstrip("/") + "/" + endpoint.lstrip("/"),
            {"Authorization": f"Bearer {key}"}, config["retry"])


def _result(response):
    status = response.status_code
    if not 200 <= status < 300:
        return None, InferenceError("post_json", status, status == 429 or 500 <= status < 600)
    try:
        data = response.json()
    except ValueError:
        data = None
    if not isinstance(data, dict):
        return None, InferenceError("post_json", status)
    return data, None


def _delay(policy, attempt, response):
    delay = min(policy["initial_delay"] * 2 ** attempt, policy["max_delay"])
    if response is not None:
        value = response.headers.get("Retry-After")
        if value:
            try:
                delay = float(value)
            except ValueError:
                try:
                    date = parsedate_to_datetime(value)
                    if date.tzinfo is None:
                        date = date.replace(tzinfo=timezone.utc)
                    delay = (date - datetime.now(timezone.utc)).total_seconds()
                except (ValueError, TypeError, OverflowError):
                    pass
    # NaN is invalid; infinity is safely capped.
    if delay != delay:
        delay = policy["initial_delay"] * 2 ** attempt
    return max(0, min(delay, policy["max_delay"]))


def post_json(endpoint: str, payload: dict, *, timeout: float) -> dict:
    """POST a JSON object; retry network failures, 429 and 5xx only."""
    url, headers, policy = _request(endpoint)
    with httpx.Client(follow_redirects=False) as client:
        for attempt in range(policy["max_attempts"]):
            response = None
            try:
                response = client.post(url, json=payload, headers=headers, timeout=timeout)
            except httpx.RequestError:
                error = InferenceError("post_json", retryable=True)
            else:
                data, error = _result(response)
                if error is None:
                    return data
            if not error.retryable or attempt + 1 >= policy["max_attempts"]:
                raise error
            time.sleep(_delay(policy, attempt, response))
    raise InferenceError("post_json")


async def apost_json(endpoint: str, payload: dict, *, timeout: float) -> dict:
    """Async counterpart using nonblocking retry waits."""
    url, headers, policy = _request(endpoint)
    async with httpx.AsyncClient(follow_redirects=False) as client:
        for attempt in range(policy["max_attempts"]):
            response = None
            try:
                response = await client.post(url, json=payload, headers=headers, timeout=timeout)
            except httpx.RequestError:
                error = InferenceError("post_json", retryable=True)
            else:
                data, error = _result(response)
                if error is None:
                    return data
            if not error.retryable or attempt + 1 >= policy["max_attempts"]:
                raise error
            await asyncio.sleep(_delay(policy, attempt, response))
    raise InferenceError("post_json")
