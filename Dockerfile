# Slim CPU image: FastAPI + fastembed (ONNX) + llama.cpp. Used for the Hugging Face Space (Docker SDK, port 7860)
# and for local `docker compose up`. No torch, no GPU.
#
#   docker build -t mhrag .
#   docker run -p 7860:7860 -e GROQ_API_KEY=... mhrag          # API model; without a key -> local GGUF on CPU

# ---------------------------------------------------------------- stage 1: build wheels (llama.cpp needs a compiler)
FROM python:3.11-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_BLAS=OFF"
RUN pip install --upgrade pip wheel \
    && pip wheel --wheel-dir /wheels -r /tmp/requirements.txt

# ---------------------------------------------------------------- stage 2: runtime
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 user
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index /wheels/* && rm -rf /wheels

USER user
ENV HOME=/home/user \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    FASTEMBED_CACHE_PATH=/home/user/.cache/fastembed
WORKDIR /home/user/app
COPY --chown=user . .

ENV MHRAG_EMBEDDER_BACKEND=fastembed \
    MHRAG_LLAMACPP_GPU_LAYERS=0 \
    MHRAG_LLAMACPP_CTX=4096 \
    MHRAG_LLM__MODEL=auto \
    MHRAG_LLM__SERVED_MODELS='["qwen2.5-1.5b-gguf","llama-3.2-1b-gguf","llama-3.2-3b-gguf","gemma-2-2b-gguf","phi-3.5-mini-gguf","qwen3-4b-gguf"]' \
    MHRAG_SERVER__MAX_CONCURRENT_GENERATIONS=1 \
    MHRAG_SERVER__PORT=7860

# The prebuilt index (artifacts/index/) is normally shipped by CI; build it here only if it is missing and the
# raw data is present. Then fetch the ONNX embedder/reranker and the fallback GGUF so cold starts are quick.
ARG PREFETCH_MODELS=1
RUN if [ ! -f artifacts/index/bge-small-c256/manifest.json ] && [ -f data/raw_data.zip ]; then \
        python -m scripts.build_index --backend fastembed; fi \
    && if [ "$PREFETCH_MODELS" = "1" ]; then python -m scripts.prefetch_models; fi

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD curl -fs http://localhost:7860/api/health || exit 1
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]
