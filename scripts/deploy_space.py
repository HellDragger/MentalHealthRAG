"""Upload the app to a Hugging Face Space (Docker SDK). Used by CI and by hand.

    export HF_TOKEN=hf_...                       # write token
    python -m scripts.build_index --backend fastembed
    python -m scripts.deploy_space --space <user>/<space-name>

The Space receives the code, the prebuilt fastembed index and the small TF-IDF risk classifier. The raw datasets
(data/raw_data.zip) are NOT uploaded: they are third-party data (Mind, Kaggle, Reddit/Twitter) that we do not
redistribute, and the prebuilt index makes them unnecessary at serve time.
"""

from __future__ import annotations

import argparse
import os
import sys

from mhrag.config import PROJECT_ROOT, get_settings
from mhrag.index.builder import index_dir
from mhrag.index.store import read_manifest

IGNORE = [
    ".git/*", ".github/*", ".venv/*", "**/__pycache__/*", ".pytest_cache/*", ".ruff_cache/*", ".claude/*",
    ".env", "data/*", "paper/*", "notebooks/*", "results/generation/*", "results/logs/*", "eval/human_eval/out/*",
    "artifacts/risk_classifier/transformer/*", "*.gguf", "Blank diagram.png",
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", default=os.environ.get("HF_SPACE_ID"), help="<user or org>/<space name>")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args(argv)
    token = os.environ.get("HF_TOKEN")
    if not token or not args.space:
        print("Set HF_TOKEN and --space (or HF_SPACE_ID).", file=sys.stderr)
        return 2

    s = get_settings()
    idx = index_dir(s)
    m = read_manifest(idx)
    if not m:
        print(f"No index at {idx}. Run: python -m scripts.build_index --backend fastembed", file=sys.stderr)
        return 2
    if m["embedder"]["backend"] != "fastembed":
        print(f"warning: index built with {m['embedder']['backend']}; the Space queries with fastembed. "
              "Rebuild with --backend fastembed --force for an exact match.", file=sys.stderr)
    # only ship the default index
    ignore = IGNORE + ["artifacts/index/.build.lock"] + [
        f"artifacts/index/{p.name}/*" for p in (s.artifacts_dir / "index").iterdir() if p != idx]

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    api.upload_folder(repo_id=args.space, repo_type="space", folder_path=str(PROJECT_ROOT),
                      ignore_patterns=ignore, commit_message="Deploy from GitHub")
    print(f"Uploaded to https://huggingface.co/spaces/{args.space}")
    print("Add GROQ_API_KEY under Settings -> Variables and secrets for the fast API model (optional).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
