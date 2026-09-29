import json

import pytest
from fastapi.testclient import TestClient

from mhrag.llm.mock import FAIL_TRIGGER
from mhrag.runtime import build_runtime


@pytest.fixture()
def runtime(built_index):
    settings, _, _ = built_index
    return build_runtime(settings, warmup=False)


def test_full_pipeline_grounds_and_cites(runtime):
    out = runtime.pipeline.answer("How can I improve my sleep routine?")
    assert out["generated"] and out["gate"] == "none"
    assert out["sources"] and out["sources"][0]["title"]
    assert "[1]" in out["answer"]
    assert "Context:" in runtime.models.get("mock").calls[-1][-1]["content"]
    assert out["timing"]["ttft_ms"] is not None


def test_crisis_skips_generation(runtime):
    mock = runtime.models.get("mock")
    n_calls = len(mock.calls)
    out = runtime.pipeline.answer("I want to kill myself tonight", region="UK")
    assert out["gate"] == "crisis" and not out["generated"]
    assert "116 123" in out["answer"]
    assert len(mock.calls) == n_calls  # the LLM was never called


def test_naive_and_no_rag_profiles(runtime):
    naive = runtime.pipeline.answer("anger", profile="naive")
    assert len(naive["sources"]) == 2 and naive["gate"] is None
    no_rag = runtime.pipeline.answer("What is anger?", profile="no_rag")
    assert no_rag["sources"] == []


def test_history_is_trimmed(runtime):
    hist = [{"role": "user", "content": f"q{i}"} if i % 2 == 0 else {"role": "assistant", "content": f"a{i}"}
            for i in range(40)]
    runtime.pipeline.answer("What is anger?", history=hist)
    sent = runtime.models.get("mock").calls[-1]
    assert len(sent) <= 2 + 2 * runtime.settings.llm.history_turns


# ------------------------------------------------------------------------------ server
@pytest.fixture()
def client(built_index, monkeypatch):
    settings, _, _ = built_index
    import server.app as srv

    monkeypatch.setattr(srv, "settings", settings)
    monkeypatch.setattr("mhrag.runtime.get_settings", lambda: settings)
    with TestClient(srv.app) as c:
        yield c


def events(resp):
    out = []
    for block in resp.text.strip().split("\n\n"):
        ev = {ln.split(":", 1)[0]: ln.split(":", 1)[1].strip() for ln in block.split("\n")}
        out.append((ev["event"], json.loads(ev["data"])))
    return out


def test_health_and_models(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["index"]["chunks"] > 0
    assert client.get("/api/models").json()["default"] == "mock"
    assert client.get("/api/helplines?region=US").json()["helplines"][0]["region"] == "US"


def test_chat_streams_sse(client):
    r = client.post("/api/chat", json={"message": "How can I improve my sleep?"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    kinds = [k for k, _ in events(r)]
    assert kinds[0] == "gate" and "sources" in kinds and "token" in kinds and kinds[-1] == "done"
    done = events(r)[-1][1]
    assert "context" not in done and done["request_id"]


def test_backend_error_is_generic(client):
    r = client.post("/api/chat", json={"message": f"What is anger? {FAIL_TRIGGER}"})
    kind, data = events(r)[-1]
    assert kind == "error" and "mock backend" not in data["message"] and data["request_id"]


def test_validation_errors(client):
    assert client.post("/api/chat", json={"message": ""}).status_code == 422
    assert client.post("/api/chat", json={"message": "x" * 5000}).status_code == 422
    assert client.post("/api/chat", content=b"not json", headers={"content-type": "text/plain"}).status_code == 422


def test_ui_served_with_security_headers(client):
    r = client.get("/")
    assert r.status_code == 200 and "Not a substitute for professional care" in r.text
    assert "script-src 'self'" in r.headers["content-security-policy"]
    assert client.get("/static/app.js").status_code == 200


def test_refuses_without_index(tmp_path, monkeypatch, settings):
    import asyncio

    import server.app as srv

    settings.paths.artifacts = str(tmp_path / "empty")
    monkeypatch.setattr(srv, "settings", settings)
    monkeypatch.setattr("mhrag.runtime.get_settings", lambda: settings)

    async def start():
        async with srv.lifespan(srv.app):
            pass

    with pytest.raises(SystemExit):
        asyncio.run(start())


def test_length_truncated_answer_is_trimmed_and_flagged(runtime):
    params = runtime.pipeline.params(max_new_tokens=6)
    out = runtime.pipeline.answer("How can I improve my sleep routine?", params=params)
    assert out["truncated"] is True
    assert "Ask me to continue" in out["answer"]


def test_trim_to_boundary():
    from mhrag.prompts import trim_to_boundary

    assert trim_to_boundary("First point. Second point is cut mid") == "First point."
    assert trim_to_boundary("Intro:\n- one [1].\n- two is cut") == "Intro:\n- one [1]."


def test_api_key_models_are_hidden_unless_allowed(built_index, monkeypatch):
    import pytest

    from mhrag.llm.registry import load_catalog, needs_api_key
    from mhrag.runtime import build_runtime

    monkeypatch.setenv("GROQ_API_KEY", "set")  # even with a key, API models stay hidden by default
    settings, _, _ = built_index
    api = {k for k, spec in load_catalog().items() if needs_api_key(spec)}
    assert {"gpt-oss-120b-groq", "llama-3.3-70b-groq", "hf-router-custom"} <= api and "vllm-local" not in api

    settings.llm.served_models = ["*", "gpt-oss-120b-groq"]
    rt = build_runtime(settings, warmup=False)
    listed = {m["key"] for m in rt.models.list()}
    assert not listed & api and "qwen2.5-1.5b-gguf" in listed
    with pytest.raises(Exception, match="not served"):
        rt.models.get("gpt-oss-120b-groq")  # also refused if the UI is bypassed

    settings.llm.served_models, settings.llm.model = ["*"], "gpt-oss-120b-groq"
    with pytest.raises(SystemExit, match="ALLOW_API_MODELS"):
        build_runtime(settings, warmup=False)

    settings.llm.model, settings.llm.allow_api_models = "mock", True
    assert api <= {m["key"] for m in build_runtime(settings, warmup=False).models.list()}
