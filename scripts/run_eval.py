"""Run an experiment described by a YAML file and write results/ + paper/tables/.

    python -m scripts.run_eval --config configs/experiments/retrieval_main.yaml
    python -m scripts.run_eval --config configs/experiments/retrieval_chunks.yaml
    python -m scripts.run_eval --config configs/experiments/generation_local.yaml
    python -m scripts.run_eval --config configs/experiments/generation_gpu.yaml     # Colab / GPU
    python -m scripts.run_eval --config configs/experiments/generation_gpu.yaml --models qwen2.5-7b-instruct
    python -m scripts.run_eval --tables-only --config ...                           # re-render LaTeX

Generation runs checkpoint every answer to results/generation/<exp>/<model>__<profile>__<dataset>.jsonl and
resume from it, so an interrupted Colab session can continue.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

from mhrag.config import PROJECT_ROOT, get_settings

log = logging.getLogger("run_eval")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tables-only", action="store_true")
    ap.add_argument("--limit", type=int, help="use only the first N queries of each dataset (smoke runs)")
    ap.add_argument("--generate-only", action="store_true",
                    help="generation only: save answers but skip scoring (e.g. one process per GPU; score afterwards "
                         "with a normal run, which reuses the saved answers)")
    ap.add_argument("--models", nargs="+", help="generation only: run just these model keys (e.g. one per Kaggle session); "
                                               "results are merged with earlier runs of the same experiment")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "sentence_transformers", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    path = Path(args.config)
    cfg = yaml.safe_load((path if path.is_absolute() else PROJECT_ROOT / path).read_text())
    s = get_settings()
    for k, v in (cfg.get("settings") or {}).items():  # experiment-level overrides, e.g. retrieval.top_k
        section, key = k.split(".", 1)
        setattr(getattr(s, section), key, v)
    out = s.results_dir / f"{cfg['kind']}_{cfg['name']}.json"

    if cfg["kind"] == "retrieval":
        from eval.report import retrieval_tables
        from eval.retrieval_eval import run_retrieval_experiment

        if not args.tables_only:
            if args.limit:
                cfg["limit"] = args.limit
            ckpt = out.with_suffix(".partial.jsonl")  # finished runs; an interrupted experiment resumes from here
            res = run_retrieval_experiment(cfg, s, checkpoint=None if args.limit else ckpt)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(res))
            ckpt.unlink(missing_ok=True)
        retrieval_tables(json.loads(out.read_text()))
    elif cfg["kind"] == "generation":
        if os.environ.get("MHRAG_JUDGE_MODEL"):  # e.g. a judge the API key has access to
            cfg["judge_model"] = os.environ["MHRAG_JUDGE_MODEL"]
        from eval.generation_eval import run_generation_experiment
        from eval.report import generation_tables

        if args.models:
            unknown = [m for m in args.models if m not in cfg["models"]]
            if unknown:
                raise SystemExit(f"--models not in the experiment config: {unknown}")
            cfg["run_models"] = args.models
        from eval.generation_eval import RateLimited

        if not args.tables_only:
            try:
                if args.generate_only:
                    from eval.generation_eval import generate

                    out_dir = s.results_dir / "generation" / cfg["name"]
                    out_dir.mkdir(parents=True, exist_ok=True)
                    generate(cfg, s, out_dir, args.limit)
                    print(f"answers saved in {out_dir} (not scored: run without --generate-only to score)")
                    return 0
                res = run_generation_experiment(cfg, s, limit=args.limit)
            except RateLimited as e:
                print(f"API quota exhausted ({e}). All answers so far are saved; run again later to continue.")
                return 75
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(res, indent=1))
        res = json.loads(out.read_text())
        # report what the config lists now (a run excluded after the fact stays in the results file but not the tables)
        res["report_models"], res["excluded_runs"] = cfg["models"], cfg.get("excluded_runs") or {}
        generation_tables(res)
        if res.get("judge_status") == "incomplete" and not args.tables_only:
            print(f"wrote {out}, but the LLM judge is incomplete (API quota); its columns stay TODO(run). "
                  "Run again later: cached verdicts are reused.")
            return 75
    else:
        raise SystemExit(f"unknown kind {cfg['kind']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
