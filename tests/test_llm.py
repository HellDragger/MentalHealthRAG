import json

import httpx
import pytest

from mhrag.llm import remote
from mhrag.llm.base import BackendError, GenerationParams, ModelSpec
from mhrag.llm.mock import FAIL_TRIGGER, MockBackend
from mhrag.llm.registry import ModelManager, availability, load_catalog
from mhrag.llm.templates import (
    llama2_prompt,
    merge_system,
    prepare_messages,
    strip_reasoning,
    template_kwargs,
)

MSGS = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hello"}]


def test_catalog_entries_are_well_formed():
    cat = load_catalog()
    required = {"mistral-7b-instruct-v0.3", "llama-3.1-8b-instruct", "llama-3.2-3b-instruct", "qwen2.5-7b-instruct",
                "qwen2.5-1.5b-instruct", "qwen3-4b", "qwen3-8b", "phi-3.5-mini-instruct", "phi-4-mini-instruct",
                "mentallama-chat-7b", "llama-3.3-70b-groq", "gemma-2-2b-it", "gemma-3-12b-it", "mock"}
    assert required <= set(cat)
    for k, spec in cat.items():
        assert spec.backend in {"hf", "llamacpp", "ollama", "openai_compatible", "mock"}, k
        if spec.backend == "hf":
            assert "hf_id" in spec.raw and "license" in spec.raw and "gated" in spec.raw, k
        if spec.backend == "openai_compatible":
            assert "api_key_env" in spec.raw, k
            assert "api_key" not in spec.raw, "API keys must never be hard-coded"


def test_gemma_has_no_system_role():
    out = prepare_messages(MSGS, "gemma")
    assert [m["role"] for m in out] == ["user"] and out[0]["content"].startswith("SYS")
    assert prepare_messages(MSGS, "qwen2")[0]["role"] == "system"


def test_merge_system_without_user():
    assert merge_system([{"role": "system", "content": "S"}]) == [{"role": "user", "content": "S"}]


def test_llama2_prompt_format():
    p = llama2_prompt(MSGS + [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "q2"}])
    assert p.startswith("<s>[INST] <<SYS>>\nSYS\n<</SYS>>\n\nhello [/INST] hi </s><s>[INST] q2 [/INST]")


def test_qwen3_non_thinking():
    assert template_kwargs("qwen3") == {"enable_thinking": False}
    assert strip_reasoning("<think>hmm</think>Answer") == "Answer"


def test_sampling_flag_follows_temperature():
    assert not GenerationParams(temperature=0).do_sample
    assert GenerationParams(temperature=0.7).do_sample


def test_mock_backend_streams_and_fails_on_request():
    b = MockBackend(ModelSpec("mock", "mock"), delay_s=0)
    out = b.generate([{"role": "user", "content": "Context:\n[1] T\nSleep helps.\n\nQuestion: ?"}], GenerationParams())
    assert "[1]" in out
    with pytest.raises(BackendError):
        b.generate([{"role": "user", "content": FAIL_TRIGGER}], GenerationParams())


def _sse_body(chunks, usage=None):
    lines = [f"data: {json.dumps({'choices': [{'delta': {'content': c}}]})}" for c in chunks]
    if usage:
        lines.append(f"data: {json.dumps({'choices': [], 'usage': usage})}")
    lines.append("data: [DONE]")
    return "\n\n".join(lines) + "\n\n"


@pytest.fixture()
def fake_http(monkeypatch):
    calls = []

    def handler(request: httpx.Request):
        calls.append(request)
        if request.headers.get("authorization") != "Bearer sk-test":
            return httpx.Response(401, json={"error": "bad key"})
        return httpx.Response(200, text=_sse_body(["Hel", "lo"], {"prompt_tokens": 5, "completion_tokens": 2}),
                              headers={"content-type": "text/event-stream"})

    real = httpx.Client

    def client(*a, **kw):
        kw["transport"] = httpx.MockTransport(handler)
        return real(*a, **kw)

    monkeypatch.setattr(remote.httpx, "Client", client)
    return calls


def test_openai_compatible_streams_and_reads_key_from_env(fake_http, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "sk-test")
    spec = load_catalog()["llama-3.3-70b-groq"]
    b = remote.OpenAICompatibleBackend(spec)
    assert b.generate(MSGS, GenerationParams(temperature=0)) == "Hello"
    body = json.loads(fake_http[0].content)
    assert body["model"] == "llama-3.3-70b-versatile" and body["stream"] is True
    assert b.last_usage["completion_tokens"] == 2


def test_openai_compatible_bad_key_is_backend_error(fake_http, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "wrong")
    b = remote.OpenAICompatibleBackend(load_catalog()["llama-3.3-70b-groq"])
    with pytest.raises(BackendError) as e:
        b.generate(MSGS, GenerationParams())
    assert "wrong" not in str(e.value)  # never leak the key


def test_missing_api_key_marks_model_unavailable(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    ok, why = availability(load_catalog()["llama-3.3-70b-groq"])
    assert not ok and "GROQ_API_KEY" in why


def test_model_manager_serves_only_configured_models():
    mm = ModelManager(load_catalog(), ["mock"], "mock")
    assert mm.get("mock") is mm.get()  # loaded once
    with pytest.raises(BackendError):
        mm.get("qwen3-8b")
    assert mm.list()[0]["key"] == "mock"


def test_local_backend_generation_is_serialised():
    import threading
    import time

    from mhrag.llm.base import Backend

    class Slow(Backend):
        active = 0
        peak = 0

        def _stream(self, messages, params):
            Slow.active += 1
            Slow.peak = max(Slow.peak, Slow.active)
            time.sleep(0.05)
            Slow.active -= 1
            yield "x"

        def stream(self, messages, params):
            with self.generation_lock:
                yield from self._stream(messages, params)

    b = Slow(ModelSpec("slow", "hf"))
    ts = [threading.Thread(target=lambda: b.generate(MSGS, GenerationParams())) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert Slow.peak == 1


def test_serve_all_lists_whole_catalogue_grouped_with_reasons(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    cat = load_catalog()
    mm = ModelManager(cat, ["*"], "mock")
    listed = mm.list()
    assert {m["key"] for m in listed} == set(cat)
    assert listed[0]["key"] == "mock" and listed[0]["default"]
    groq = next(m for m in listed if m["key"] == "llama-3.3-70b-groq")
    assert not groq["available"] and "GROQ_API_KEY" in groq["status"] and groq["group"] == "api"
    assert next(m for m in listed if m["key"] == "gpt2")["group"] == "baseline"
    assert next(m for m in listed if m["key"] == "tinyllama-1.1b-chat")["group"] != "baseline"
    for m in listed:
        assert m["group_label"] and (m["available"] or m["status"])


def test_memory_check_blocks_models_that_do_not_fit(monkeypatch):
    from mhrag.llm import registry

    monkeypatch.setattr(registry, "device_memory_gb", lambda: (8.0, "RAM"))
    spec = load_catalog()["llama-3.1-8b-gguf"]
    ok, why = registry.availability(spec)
    if registry.importlib.util.find_spec("llama_cpp") is not None:
        assert ok is True or "needs" in why  # 8B Q4 ~5.6 GB fits in 8 GB * 0.8
    spec7 = load_catalog()["qwen2.5-7b-instruct"]
    if registry.importlib.util.find_spec("torch") is not None:
        ok, why = registry.availability(spec7)
        assert not ok and "needs" in why


def test_model_not_found_lists_models_the_key_can_use(monkeypatch):
    import httpx

    import mhrag.llm.remote as remote
    from mhrag.llm.base import BackendError, GenerationParams, ModelSpec

    def handler(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "openai/gpt-oss-120b"}, {"id": "llama-3.1-8b-instant"}]})
        return httpx.Response(404, json={"error": {"message": "The model `x` does not exist or you do not have access"}})

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client
    monkeypatch.setattr(remote.httpx, "Client", lambda **kw: real_client(transport=transport, **kw))
    monkeypatch.setattr(remote.httpx, "get", lambda url, **kw: real_client(transport=transport).get(url, **kw))
    monkeypatch.setenv("TEST_KEY", "k")
    spec = ModelSpec(key="judge", backend="openai_compatible", label="j", family="generic", context=8192,
                     raw={"base_url": "https://api.example.com/v1", "api_model": "x", "api_key_env": "TEST_KEY"})
    b = remote.OpenAICompatibleBackend(spec, max_retries=0)
    with pytest.raises(BackendError) as e:
        b.generate([{"role": "user", "content": "hi"}], GenerationParams(max_new_tokens=3, temperature=0))
    assert "HTTP 404" in str(e.value) and "llama-3.1-8b-instant, openai/gpt-oss-120b" in str(e.value)


def test_per_minute_rate_limit_is_waited_out_but_daily_quota_fails(monkeypatch):
    import httpx

    import mhrag.llm.remote as remote
    from mhrag.llm.base import BackendError, GenerationParams, ModelSpec

    state = {"n": 0, "retry_after": "2"}
    sse = 'data: {"choices":[{"delta":{"content":"OK"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'

    def handler(request):
        state["n"] += 1
        if state["n"] <= 3:
            return httpx.Response(429, headers={"retry-after": state["retry_after"]},
                                  json={"error": {"message": "Rate limit reached"}})
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client
    sleeps = []
    monkeypatch.setattr(remote.httpx, "Client", lambda **kw: real_client(transport=transport, **kw))
    monkeypatch.setattr(remote.time, "sleep", sleeps.append)
    monkeypatch.setenv("TEST_KEY", "k")
    spec = ModelSpec(key="api", backend="openai_compatible", label="a", family="generic", context=8192,
                     raw={"base_url": "https://api.example.com/v1", "api_model": "x", "api_key_env": "TEST_KEY",
                          "request_params": {"reasoning_effort": "low"}})
    b = remote.OpenAICompatibleBackend(spec, max_retries=2)
    assert b.generate([{"role": "user", "content": "hi"}], GenerationParams(max_new_tokens=3)) == "OK"
    assert sleeps == [2.5, 2.5, 2.5]  # three per-minute waits, not counted as failed attempts

    state.update(n=0, retry_after="3600")  # a daily quota: give up quickly instead of sleeping for an hour
    sleeps.clear()
    with pytest.raises(BackendError, match="HTTP 429"):
        b.generate([{"role": "user", "content": "hi"}], GenerationParams(max_new_tokens=3))
    assert max(sleeps) <= 10
