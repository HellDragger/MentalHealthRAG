"""Build the retrieval index (idempotent).

    python -m scripts.build_index                      # default embedder + chunk size from configs/default.yaml
    python -m scripts.build_index --embedder minilm --chunk-tokens 128
    python -m scripts.build_index --grid               # every embedder x {128,256,512} (for experiments)
    python -m scripts.build_index --backend fastembed  # ONNX, no torch (CPU deployment)
    python -m scripts.build_index --force              # rebuild even if nothing changed

Writes results/ingest_stats.json for the default configuration.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from mhrag.config import get_settings
from mhrag.index.builder import build_index


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedder", help="embedder key from configs/default.yaml")
    ap.add_argument("--chunk-tokens", type=int)
    ap.add_argument("--backend", choices=["auto", "sentence_transformers", "fastembed", "hashing"])
    ap.add_argument("--variant", default="", help="suffix for ablation indexes, e.g. 'mindonly'")
    ap.add_argument("--grid", action="store_true", help="build every embedder x chunk size")
    ap.add_argument("--embedders", nargs="*", help="restrict --grid to these embedders")
    ap.add_argument("--chunk-sizes", nargs="*", type=int, default=[128, 256, 512])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    s = get_settings()
    jobs = []
    if args.grid:
        for e in args.embedders or list(s.embedders):
            for c in args.chunk_sizes:
                jobs.append((e, c))
    else:
        jobs.append((args.embedder or s.index.embedder, args.chunk_tokens or s.chunking.chunk_tokens))

    for emb, ct in jobs:
        path, stats, built = build_index(s, emb, ct, args.variant, args.force, args.backend)
        print(f"{'built' if built else 'up-to-date'}: {path}  ({stats.get('chunking', {}).get('chunks', '?')} chunks)")
        is_default = (emb, ct) == (s.index.embedder, s.chunking.chunk_tokens) and not args.variant
        if built and is_default:
            s.results_dir.mkdir(parents=True, exist_ok=True)
            (s.results_dir / "ingest_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
