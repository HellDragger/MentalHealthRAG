"""HTTP backends: any OpenAI-compatible server (Groq, OpenRouter, Together, HF router, vLLM, OpenAI) and Ollama.

API keys are read from the environment variable named in configs/models.yaml (`api_key_env`) and are never
logged or returned to clients.
"""

from __future__ import annotations

import json
import logging
import os
import time

import httpx

from mhrag.llm.base import Backend, BackendError, GenerationParams, Message
from mhrag.llm.templates import prepare_messages, strip_reasoning

log = logging.getLogger(__name__)


def _env_or(raw: dict, key: str) -> str | None:
    env = raw.get(f"{key}_env")
    if env and os.environ.get(env):
        return os.environ[env]
    return raw.get(key)


def _retry_after(r) -> float | None:
    try:
        return float(r.headers["retry-after"])
    except (KeyError, ValueError):
        return None


def _api_error(r) -> str:
    """The provider's error message (e.g. "model not found"), never the request content."""
    try:
        err = r.json().get("error")
        msg = err.get("message") if isinstance(err, dict) else err
        return f" ({str(msg)[:200]})" if msg else ""
    except Exception:
        return ""


class OpenAICompatibleBackend(Backend):
    name = "openai_compatible"

    def __init__(self, spec, timeout_s: float = 120.0, max_retries: int = 2):
        super().__init__(spec)
        raw = spec.raw
        self.base_url = (_env_or(raw, "base_url") or "").rstrip("/")
        self.model = _env_or(raw, "api_model")
        key_env = raw.get("api_key_env")
        self.api_key = os.environ.get(key_env, "") if key_env else ""
        if not self.base_url or not self.model:
            raise BackendError(f"{spec.key}: base_url/api_model not configured")
        if not self.api_key and not raw.get("api_key_optional"):
            raise BackendError(f"{spec.key}: environment variable {key_env} is not set")
        self.timeout = httpx.Timeout(timeout_s, connect=10.0)
        self.max_retries = max_retries

    def _available_models(self) -> str:
        """On 'model not found', list the models this key can use (GET /models), to tell a wrong model id from a
        key whose organisation, project or plan does not include the model."""
        try:
            r = httpx.get(f"{self.base_url}/models", headers=self._headers(), timeout=10.0)
            ids = sorted(m["id"] for m in r.json().get("data", []))
        except Exception:
            return ""
        return f". This key can use: {', '.join(ids)}" if ids else ""

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def stream(self, messages: list[Message], params: GenerationParams):
        body = {
            "model": self.model,
            "messages": prepare_messages(messages, self.spec.family),
            "stream": True,
            "max_tokens": params.max_new_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "stream_options": {"include_usage": True},
        }
        if params.seed is not None:
            body["seed"] = params.seed
        body.update(self.spec.raw.get("request_params") or {})  # provider-specific, e.g. reasoning_effort
        url = f"{self.base_url}/chat/completions"
        attempt, rate_waits = -1, 0
        self.last_rate_wait_s = 0.0  # time spent waiting out per-minute rate limits in this request
        while True:
            attempt += 1
            emitted = False
            try:
                with httpx.Client(timeout=self.timeout) as client, client.stream(
                    "POST", url, headers=self._headers(), json=body
                ) as r:
                    if r.status_code == 400 and "stream_options" in body:
                        body.pop("stream_options")  # some servers reject it; retry without
                        raise httpx.HTTPStatusError("retry without stream_options", request=r.request, response=r)
                    wait = _retry_after(r)
                    if r.status_code == 429 and wait is not None and wait <= 120 and rate_waits < 8:
                        # a per-minute limit: wait it out (a daily quota asks for a much longer wait and fails below)
                        rate_waits += 1
                        attempt -= 1
                        log.info("%s: rate limited; waiting %.0f s", self.spec.key, wait + 0.5)
                        time.sleep(wait + 0.5)
                        self.last_rate_wait_s += wait + 0.5
                        continue
                    if r.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                        wait = float(r.headers.get("retry-after", 2 ** attempt))
                        time.sleep(min(wait, 10))
                        continue
                    if r.status_code >= 400:
                        r.read()
                        hint = self._available_models() if r.status_code == 404 else ""
                        raise BackendError(f"{self.spec.key}: HTTP {r.status_code}{_api_error(r)}{hint}")
                    for line in r.iter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        ev = json.loads(data)
                        if ev.get("usage"):
                            self.last_usage = ev["usage"]
                        for ch in ev.get("choices") or []:
                            if ch.get("finish_reason"):
                                self.last_finish_reason = ch["finish_reason"]
                            delta = (ch.get("delta") or {}).get("content")
                            if delta:
                                emitted = True
                                yield strip_reasoning(delta) if "<think>" in delta else delta
                    return
            except httpx.HTTPStatusError:
                if attempt < self.max_retries:
                    continue
                raise BackendError(f"{self.spec.key}: request rejected") from None
            except (httpx.TransportError, json.JSONDecodeError) as e:
                if emitted or attempt >= self.max_retries:
                    raise BackendError(f"{self.spec.key}: {type(e).__name__}") from None
                time.sleep(1 + attempt)


class OllamaBackend(Backend):
    name = "ollama"

    def __init__(self, spec, timeout_s: float = 120.0):
        super().__init__(spec)
        raw = spec.raw
        self.base_url = (_env_or(raw, "base_url") or "http://localhost:11434").rstrip("/")
        self.model = raw["ollama_tag"]
        self.timeout = httpx.Timeout(timeout_s, connect=5.0)

    def stream(self, messages: list[Message], params: GenerationParams):
        body = {
            "model": self.model,
            "messages": prepare_messages(messages, self.spec.family),
            "stream": True,
            "options": {
                "temperature": params.temperature,
                "top_p": params.top_p,
                "num_predict": params.max_new_tokens,
                "repeat_penalty": params.repetition_penalty,
                **({"seed": params.seed} if params.seed is not None else {}),
            },
        }
        try:
            with httpx.Client(timeout=self.timeout) as client, client.stream(
                "POST", f"{self.base_url}/api/chat", json=body
            ) as r:
                if r.status_code >= 400:
                    r.read()
                    raise BackendError(f"ollama HTTP {r.status_code}")
                for line in r.iter_lines():
                    if not line:
                        continue
                    ev = json.loads(line)
                    piece = (ev.get("message") or {}).get("content")
                    if piece:
                        yield piece
                    if ev.get("done"):
                        self.last_finish_reason = ev.get("done_reason")
                        self.last_usage = {
                            "prompt_tokens": ev.get("prompt_eval_count"),
                            "completion_tokens": ev.get("eval_count"),
                        }
                        break
        except httpx.TransportError as e:
            raise BackendError(f"ollama unreachable at {self.base_url}: {type(e).__name__}") from None
