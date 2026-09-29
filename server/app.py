"""FastAPI server.

    uvicorn server.app:app --host 0.0.0.0 --port 8000

Endpoints
- GET  /api/health     index + model status
- GET  /api/models     models this server can serve (UI model selector)
- GET  /api/helplines  helplines for a region
- POST /api/chat       Server-Sent Events: gate, sources, token*, [replace|append], done | error
- GET  /               the web UI (web/)

Privacy: message content is never logged unless MHRAG_SERVER__LOG_CONTENT=true. Logs contain a request id,
timings, the gate label and the model key only.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from mhrag.config import PROJECT_ROOT, get_settings
from mhrag.index.store import IndexError_
from mhrag.llm.base import BackendError
from mhrag.safety.crisis import helplines, regions

log = logging.getLogger("mhrag.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
for _noisy in ("httpx", "sentence_transformers", "huggingface_hub", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
WEB_DIR = PROJECT_ROOT / "web"
settings = get_settings()


def _client_ip(request: Request) -> str:
    # Behind a proxy (HF Spaces, Render) run uvicorn with --proxy-headers so request.client is the real client.
    # X-Forwarded-For is not parsed here: its left-most entry is client-controlled and would allow evading the limit.
    return get_remote_address(request)


limiter = Limiter(key_func=_client_ip)


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[Turn] = Field(default_factory=list, max_length=40)
    model: str | None = None
    region: str | None = Field(default=None, max_length=8)

    @field_validator("message")
    @classmethod
    def _limit(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message is empty")
        if len(v) > settings.server.max_message_chars:
            raise ValueError(f"message longer than {settings.server.max_message_chars} characters")
        return v


@asynccontextmanager
async def lifespan(app: FastAPI):
    from mhrag.runtime import build_runtime

    try:
        rt = build_runtime(settings)
    except IndexError_ as e:
        # Refuse to start with a clear, actionable message (CHANGELOG B2).
        log.error("\n\n*** Cannot start: %s\n", e)
        raise SystemExit(2) from None
    app.state.rt = rt
    app.state.sem = threading.BoundedSemaphore(settings.server.max_concurrent_generations)
    m = rt.index.manifest
    log.info("Ready: index=%s (%d chunks, embedder=%s), default model=%s, gate classifier=%s",
             m["index_name"], m["n_chunks"], m["embedder"]["model"], settings.llm.model,
             rt.gate.classifier.name if rt.gate.classifier else "none")
    yield


app = FastAPI(title="MHRAG", version="2.0.0", lifespan=lifespan, docs_url="/api/docs", redoc_url=None)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limited(request: Request, exc: RateLimitExceeded):
    return JSONResponse({"error": "Too many requests. Please wait a minute and try again."}, status_code=429)


@app.middleware("http")
async def _security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    resp.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
        "connect-src 'self'; frame-ancestors *",
    )
    return resp


@app.get("/api/health")
def health(request: Request):
    rt = request.app.state.rt
    m = rt.index.manifest
    return {
        "status": "ok",
        "index": {"name": m["index_name"], "chunks": m["n_chunks"], "embedder": m["embedder"]["model"],
                  "built_at": m.get("built_at"), "corpus_hash": m["corpus_hash"]},
        "retrieval_mode": settings.retrieval.mode,
        "default_model": settings.llm.model,
        "safety": {"classifier": rt.gate.classifier.name if rt.gate.classifier else None,
                   "region": settings.safety.region},
    }


@app.get("/api/models")
def list_models(request: Request):
    return {"default": settings.llm.model, "models": request.app.state.rt.models.list()}


@app.get("/api/helplines")
def get_helplines(region: str | None = None):
    return {"regions": regions(), "helplines": helplines(region or settings.safety.region)}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
@limiter.limit(settings.server.rate_limit)
def chat(request: Request, body: ChatRequest):
    rt = request.app.state.rt
    rid = uuid.uuid4().hex[:10]
    sem: threading.BoundedSemaphore = request.app.state.sem

    def gen():
        t0 = time.perf_counter()
        label, model_used = None, body.model or settings.llm.model
        # Acquire inside the generator so the slot is always released, even if the client disconnects.
        if not sem.acquire(timeout=10):
            yield _sse("error", {"message": "The server is busy. Please try again in a moment.", "request_id": rid})
            return
        try:
            if settings.server.log_content:
                log.info("[%s] message=%r", rid, body.message)
            for ev in rt.pipeline.stream(
                body.message, [t.model_dump() for t in body.history], model=body.model, region=body.region
            ):
                if ev.type == "gate":
                    label = ev.data.get("label")
                if ev.type == "done":
                    ev.data.pop("context", None)
                    ev.data.pop("raw_answer", None)
                    ev.data["request_id"] = rid
                yield _sse(ev.type, ev.data)
        except BackendError as e:
            log.warning("[%s] backend error (%s): %s", rid, model_used, e)
            yield _sse("error", {"message": "The language model is unavailable right now. Please try again, "
                                            "or pick another model.", "request_id": rid})
        except Exception:
            log.exception("[%s] unexpected error", rid)  # stack trace stays server-side (CHANGELOG B12)
            yield _sse("error", {"message": "Something went wrong on our side. Please try again.",
                                 "request_id": rid})
        finally:
            sem.release()
            log.info("[%s] model=%s gate=%s total_ms=%.0f", rid, model_used, label, (time.perf_counter() - t0) * 1000)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Request-Id": rid})


# ------------------------------------------------------------------ static UI
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(Path(WEB_DIR) / "index.html")
