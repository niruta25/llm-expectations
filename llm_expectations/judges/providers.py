"""Reaching the three kinds of judge, over one transport.

DESIGN.md §12 names httpx as the dependency for provider calls, and one
transport for all three providers is the point rather than an economy: this
library's whole job is comparing judges to each other. Two SDKs behind two
retry policies, two error taxonomies and two token-accounting paths would make
"judge-c is the pushover" a claim about client libraries as much as about
models.

Every provider returns the same ``JudgeReply``, including token counts, so cost
and latency are attributable per judge without anything special per vendor.
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..config import JudgeSpec
from .base import JudgeError, JudgeReply, JudgeRequest, Provider

__all__ = [
    "AnthropicProvider",
    "OpenAICompatibleProvider",
    "OpenAIProvider",
    "build_provider",
    "with_retries",
]

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_TIMEOUT = 60.0


def _api_key(spec: JudgeSpec) -> str | None:
    """Read the key from the environment variable the config *names*.

    The config carries the variable name because the config gets committed. If
    the variable is unset, say which one — a 401 from a vendor is a much worse
    way to learn that.
    """
    if spec.api_key_env is None:
        return None
    key = os.environ.get(spec.api_key_env)
    if not key:
        raise JudgeError(
            f"judge {spec.id!r} needs the environment variable {spec.api_key_env} and it is "
            "not set. The config names the variable; the key belongs in your environment."
        )
    return key


@dataclass
class AnthropicProvider:
    """The Messages API."""

    spec: JudgeSpec
    client: httpx.Client = field(default_factory=lambda: httpx.Client(timeout=DEFAULT_TIMEOUT))

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def model(self) -> str:
        return self.spec.model

    def complete(self, request: JudgeRequest) -> JudgeReply:
        started = time.perf_counter()
        response = _send(
            self.client,
            ANTHROPIC_URL,
            headers={
                "x-api-key": _api_key(self.spec) or "",
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            body={
                "model": self.spec.model,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "system": request.system,
                "messages": [{"role": "user", "content": request.user}],
            },
        )
        text = "".join(
            block.get("text", "")
            for block in response.get("content", [])
            if isinstance(block, Mapping) and block.get("type") == "text"
        )
        usage = response.get("usage") or {}
        return JudgeReply(
            text=text,
            model=str(response.get("model", self.spec.model)),
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            latency_ms=(time.perf_counter() - started) * 1000,
        )


@dataclass
class OpenAIProvider:
    """Chat completions, against api.openai.com."""

    spec: JudgeSpec
    client: httpx.Client = field(default_factory=lambda: httpx.Client(timeout=DEFAULT_TIMEOUT))
    url: str = OPENAI_URL

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def model(self) -> str:
        return self.spec.model

    def complete(self, request: JudgeRequest) -> JudgeReply:
        started = time.perf_counter()
        key = _api_key(self.spec)
        headers = {"content-type": "application/json"}
        if key:
            headers["authorization"] = f"Bearer {key}"
        response = _send(
            self.client,
            self.url,
            headers=headers,
            body={
                "model": self.spec.model,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "messages": [
                    {"role": "system", "content": request.system},
                    {"role": "user", "content": request.user},
                ],
            },
        )
        choices = response.get("choices") or [{}]
        message = choices[0].get("message") or {}
        usage = response.get("usage") or {}
        return JudgeReply(
            text=str(message.get("content") or ""),
            model=str(response.get("model", self.spec.model)),
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=(time.perf_counter() - started) * 1000,
        )


@dataclass
class OpenAICompatibleProvider(OpenAIProvider):
    """The same wire format, pointed at a box you run.

    vLLM, LM Studio, Ollama's compatible endpoint. Usually no key, which is why
    ``api_key_env`` is optional in the config.
    """

    def __post_init__(self) -> None:
        endpoint = (self.spec.endpoint or "").rstrip("/")
        if not endpoint:
            raise JudgeError(f"judge {self.spec.id!r} is openai_compatible but has no endpoint")
        self.url = f"{endpoint}/chat/completions"


def _send(
    client: httpx.Client, url: str, *, headers: Mapping[str, str], body: Mapping[str, Any]
) -> Mapping[str, Any]:
    try:
        response = client.post(url, headers=dict(headers), json=dict(body))
    except httpx.HTTPError as exc:
        raise JudgeError(f"{type(exc).__name__}: {exc}") from None
    if response.status_code >= 400:
        raise JudgeError(f"HTTP {response.status_code}: {response.text[:200]}")
    try:
        payload = response.json()
    except ValueError:
        raise JudgeError(f"the response was not JSON: {response.text[:200]}") from None
    if not isinstance(payload, Mapping):
        raise JudgeError("the response was JSON but not an object")
    return payload


_RETRYABLE = ("HTTP 429", "HTTP 5", "ConnectError", "ReadTimeout", "ConnectTimeout", "PoolTimeout")


def with_retries(
    provider: Provider,
    *,
    retries: int = 3,
    base_delay: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> Provider:
    """Wrap a provider so transient failures are retried with backoff.

    Retries rate limits, server errors and connection failures. A 400 or a 401
    is not retried — the same malformed request or the same missing key will
    fail the same way three more times, three times more slowly.
    """
    jitter = rng or random.Random(0)  # noqa: S311 — backoff jitter, not cryptography

    @dataclass
    class Retrying:
        inner: Provider

        @property
        def id(self) -> str:
            return self.inner.id

        @property
        def model(self) -> str:
            return self.inner.model

        def complete(self, request: JudgeRequest) -> JudgeReply:
            last: JudgeError | None = None
            for attempt in range(retries + 1):
                try:
                    return self.inner.complete(request)
                except JudgeError as exc:
                    last = exc
                    if attempt == retries or not _is_retryable(str(exc)):
                        break
                    sleep(base_delay * (2**attempt) + jitter.random() * base_delay)
            raise JudgeError(f"{last} (after {retries} retries)") from None

    return Retrying(provider)


def _is_retryable(message: str) -> bool:
    return any(marker in message for marker in _RETRYABLE)


_PROVIDERS: Mapping[str, type] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "openai_compatible": OpenAICompatibleProvider,
}


def build_provider(spec: JudgeSpec, *, client: httpx.Client | None = None) -> Provider:
    """Construct the transport for one configured judge."""
    factory = _PROVIDERS.get(spec.provider)
    if factory is None:  # pragma: no cover — config validation rejects this first
        raise JudgeError(f"judge {spec.id!r} has unknown provider {spec.provider!r}")
    provider = factory(spec) if client is None else factory(spec, client=client)
    return with_retries(provider)
