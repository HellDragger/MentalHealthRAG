#!/usr/bin/env bash
# Reproduce every result that can be computed on a laptop without a GPU or API keys (about 2-3 h on an M-series Mac).
# GPU / API experiments: notebooks/colab_experiments.ipynb.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/logs
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "index (default) + ingestion stats";     python -m scripts.build_index
log "risk classifiers";                        python -m scripts.train_risk_classifier
log "safety gate: dev set";                    python -m scripts.eval_safety --tag v2_devset
log "safety gate: held-out set";               python -m scripts.eval_safety --data eval/data/safety_prompts_heldout.jsonl --tag v2_heldout
log "evaluation sets";                         python -m scripts.make_eval_sets
log "SynthQA (Qwen2.5-1.5B GGUF generator)";  python -m scripts.make_eval_sets --skip-static --synth --synth-model qwen2.5-1.5b-gguf --synth-n 200
log "latency";                                 python -m scripts.bench_latency --all
log "retrieval: main";                         python -m scripts.run_eval --config configs/experiments/retrieval_main.yaml
log "retrieval: chunk sizes";                  python -m scripts.run_eval --config configs/experiments/retrieval_chunks.yaml
log "generation: local models";                python -m scripts.run_eval --config configs/experiments/generation_local.yaml
log "paper numbers";                           python -m scripts.paper_numbers
log "done"
