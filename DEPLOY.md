# Deploying MHRAG

There are three options. **A (Hugging Face Space)** is free and public, and it's the one the paper links to.
**B (Render / Railway)** is an alternative host for the same Docker image. **C** is running the server on your own machine or GPU box.

Every option runs the same `server.app:app`, and only environment variables change between them.

| Variable | Meaning |
|---|---|
| `MHRAG_LLM__MODEL` | Model key from `configs/models.yaml`, or `auto` (the first available model in `llm.auto_preference`; with API models disabled, the local Qwen2.5-1.5B GGUF). |
| `MHRAG_LLM__SERVED_MODELS` | JSON list of the models shown in the UI's model selector. |
| `MHRAG_LLM__ALLOW_API_MODELS` | Defaults to `false`: models that need an API key (Groq, OpenRouter, HF router) are hidden from the UI and refused by the server. Set `true`, plus the key, to offer them. |
| `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `HF_TOKEN`, … | API keys, only used when API models are allowed (and `HF_TOKEN` for gated local models). Read from the environment only, never hard-coded. |
| `MHRAG_SAFETY__REGION` | Default helpline region: `IN`, `US`, `UK` or `INTL`. |
| `MHRAG_SERVER__RATE_LIMIT` | Per-IP limit on `/api/chat`. The default is `20/minute`. |
| `MHRAG_SERVER__LOG_CONTENT` | Defaults to `false`, meaning message text is never logged. |

---

## A. Hugging Face Space (Docker SDK, free CPU)

The free "CPU basic" hardware has 2 vCPU and 16 GB RAM. The app runs local models only: Qwen2.5-1.5B-Instruct Q4_K_M
on CPU by default, with five other small GGUF models in the selector. It works, but it is slow on 2 vCPUs (expect tens of
seconds per answer). A paid CPU upgrade or GPU hardware in the Space settings makes it faster.

### A1. One-time setup
1. Create a Hugging Face account and a **write** token: <https://huggingface.co/settings/tokens>.
2. Install the tools on your machine:
   ```bash
   python3.11 -m venv .venv && source .venv/bin/activate
   pip install -e ".[cpu]"
   ```

### A2. Build the index the Space will ship, then upload
The Space runs without torch, so build the index with the ONNX (fastembed) backend. That way the queries and the index
use exactly the same runtime.
```bash
python -m scripts.build_index --backend fastembed --force
python -m scripts.train_risk_classifier --skip-transformer     # if artifacts/risk_classifier/tfidf_lr.joblib is missing
export HF_TOKEN=hf_xxx                                        # your write token
python -m scripts.deploy_space --space <your-hf-username>/mental-health-rag
```
The script creates the Space (SDK `docker`) if it doesn't exist and uploads the code, the prebuilt index and the TF-IDF
classifier. It does **not** upload `data/raw_data.zip`: that archive holds third-party data, and the Space doesn't need it.

### A3. Configure the Space
1. Open `https://huggingface.co/spaces/<you>/mental-health-rag` → **Settings** → **Variables and secrets**.
2. Optionally add the **variable** `MHRAG_SAFETY__REGION` = `IN` (or `US` / `UK`). No API key is needed.
3. The Space rebuilds automatically. The first build takes about 10–15 minutes, because it compiles llama.cpp and
   downloads the ONNX models and the 1 GB GGUF. When it's done, the app runs at `https://<you>-mental-health-rag.hf.space`.

The README's front matter (`sdk: docker`, `app_port: 7860`) is what tells Hugging Face how to run the container.

### A4. Automatic deploys from GitHub (optional)
`.github/workflows/ci.yml` runs lint, tests and a Docker build on every push. On `main` it also syncs the Space, but only
if these are set:
- Repository **secret** `HF_TOKEN` (Settings → Secrets and variables → Actions)
- Repository **variable** `HF_SPACE_ID` = `<you>/mental-health-rag`

---

## B. Render or Railway (same Docker image)

**Render**
1. Push the repository to GitHub.
2. In Render: **New → Web Service → Build and deploy from a Git repository**, then pick the repo. Render detects the `Dockerfile`.
3. Instance type: at least 2 GB RAM (a paid instance). The free 512 MB tier is too small for the local GGUF model.
4. Render sets `PORT`, so set the **Docker Command** to
   `uvicorn server.app:app --host 0.0.0.0 --port $PORT --proxy-headers --forwarded-allow-ips *`.
5. The prebuilt index isn't in git (`artifacts/index/` is ignored), so the image builds it during `docker build`
   from `data/raw_data.zip`. That archive *is* in the repository.

**Railway**
1. **New Project → Deploy from GitHub repo**. Railway also uses the `Dockerfile`.
2. Variables: `PORT=7860` (or override the start command as for Render). Give the service at least 2 GB RAM.
3. **Settings → Networking → Generate Domain**.

---

## C. Local machine

### C1. Python (fastest for development)
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[cpu,gpu,eval,dev]"          # use just ".[cpu]" for a no-torch install
python -m scripts.build_index                 # about 30 s; skipped when nothing changed
python -m scripts.train_risk_classifier --skip-transformer   # about 1 min (optional: without it the gate uses the lexicon)
MHRAG_LLM__MODEL=qwen2.5-1.5b-gguf uvicorn server.app:app --port 8000
# open http://localhost:8000
```
Try `MHRAG_LLM__MODEL=mock` to test the UI instantly without downloading a model. API models are hidden by default; to
offer them, set `MHRAG_LLM__ALLOW_API_MODELS=true` and the key (e.g. `GROQ_API_KEY=...`).

### C2. Docker
```bash
docker compose up --build                       # http://localhost:8000
docker compose --profile ollama up --build      # adds an Ollama server
docker compose exec ollama ollama pull qwen2.5:1.5b
```
Then choose the model key `ollama-qwen2.5-1.5b`: add it to `MHRAG_LLM__SERVED_MODELS` in `.env`.

### C3. Local GPU
```bash
pip install -e ".[gpu]"
MHRAG_LLM__MODEL=qwen2.5-7b-instruct uvicorn server.app:app --port 8000
MHRAG_LOAD_IN_4BIT=1 MHRAG_LLM__MODEL=mistral-7b-instruct-v0.3 uvicorn server.app:app   # 4-bit NF4 on CUDA
```
Gated models (Llama, Gemma) need `HF_TOKEN`, and you must accept the licence on the model page first.

---

## Operational notes
- **Privacy.** The server logs a request id, timings, the safety-gate label and the model key. It never logs message text,
  unless you set `MHRAG_SERVER__LOG_CONTENT=true`. Conversation history lives in the user's browser tab (`sessionStorage`).
- **Rate limiting** is per IP and respects `X-Forwarded-For` behind the Space or Render proxy.
- **Helplines** are in `configs/helplines.yaml`, and each number cites its official source. Re-check them before
  every public deployment.
- **Not a medical device.** The UI shows a permanent disclaimer. Don't advertise the deployment as therapy or clinical advice.
