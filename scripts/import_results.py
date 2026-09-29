"""Bring the results of a Kaggle/Colab run into results/, then re-render the paper tables and numbers.

    python -m scripts.import_results ~/Downloads/mhrag_results.zip     # the notebook's output zip
    python -m scripts.import_results path/to/mhrag_results             # or an unzipped folder
    python -m scripts.import_results ... --dry-run                     # only list what would change

Only experiments the run finished (a marker in _checkpoints/) are imported, so a partial or stale copy never replaces
newer local results. latency.json is merged per configuration: new measurements are added and a pending entry never
replaces a measured one. Checkpoints, partial files and the notebook's copies of paper files are skipped; the tables
and numbers are regenerated here instead.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from mhrag.config import PROJECT_ROOT

RESULTS = PROJECT_ROOT / "results"
# notebook step -> the files and folders it produces (relative to the results folder)
STEP_OUTPUTS = {
    "classifier": ["risk_classifier.json", "safety_gate_v2_devset.json", "safety_gate_v2_heldout.json"],
    "retrieval_main": ["retrieval_main.json"],
    "retrieval_chunks": ["retrieval_chunks.json"],
    "generation_local": ["generation_local.json", "generation/local"],
    "generation_gpu": ["generation_gpu.json", "generation/gpu"],
}
EXPERIMENTS = ["retrieval_main", "retrieval_chunks", "generation_local", "generation_gpu"]


def find_root(src: Path) -> Path:
    """The folder that holds the results files (the zip's root, or a nested mhrag_results/)."""
    for cand in [src, src / "mhrag_results", *src.glob("*/mhrag_results")]:
        if (cand / "_checkpoints").is_dir() or list(cand.glob("*.json")):
            return cand
    raise SystemExit(f"no results found in {src}")


def _measured(entry) -> bool:
    """A real measurement from the current benchmark. Rows without `rate_limit_wait_s` come from the older benchmark,
    which split models across two GPUs and did not detect API throttling, so they are not imported."""
    if not (isinstance(entry, dict) and "summary" in entry):
        return False
    rows = entry.get("rows") or []
    return not rows or all("rate_limit_wait_s" in r for r in rows)


def merge_latency(src: Path, dst: Path, dry: bool) -> list[str]:
    if not src.exists():
        return []
    new = json.loads(src.read_text())
    cur = json.loads(dst.read_text()) if dst.exists() else {}
    changed = [k for k, v in new.items() if _measured(v) and v != cur.get(k)]
    if changed and not dry:
        cur.update({k: new[k] for k in changed})
        dst.write_text(json.dumps(cur, indent=2))
    return [f"latency.json: {k}" for k in changed]


def copy_path(src: Path, dst: Path, dry: bool) -> list[str]:
    """Copy a file or folder; returns the files that are new or different."""
    files = [src] if src.is_file() else [p for p in src.rglob("*") if p.is_file() and p.name != ".DS_Store"]
    changed = []
    for f in files:
        target = dst if src.is_file() else dst / f.relative_to(src)
        if target.exists() and filecmp.cmp(f, target, shallow=False):
            continue
        changed.append(str(target.relative_to(RESULTS)))
        if not dry:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
    return changed


def copy_path_differs(root: Path, outputs: list[str]) -> bool:
    return any(copy_path(root / o, RESULTS / o, dry=True) for o in outputs)


def rerun_since(root: Path, marker: Path) -> bool:
    """The step's log shows a session that started after the marker was written, i.e. the step ran again (for example
    after a config change) and did not finish: the marker is stale."""
    import datetime as dt
    import re

    log = root / "logs" / f"kaggle_{marker.stem}.log"
    try:
        finished = dt.datetime.fromisoformat(json.loads(marker.read_text())["finished"])
        starts = re.findall(r"===== session started (\d{4}-\d\d-\d\d \d\d:\d\d)", log.read_text(errors="ignore"))
    except (OSError, KeyError, ValueError):
        return False
    return any(dt.datetime.strptime(t, "%Y-%m-%d %H:%M") > finished for t in starts)


def result_is_current(root: Path, step: str) -> bool:
    """The step's result was produced with today's experiment config and is complete (e.g. its LLM judge sample)."""
    res_path, cfg_path = root / f"{step}.json", PROJECT_ROOT / "configs" / "experiments" / f"{step}.yaml"
    if not (res_path.exists() and cfg_path.exists()):
        return True
    import yaml

    res = json.loads(res_path.read_text())
    used, now = res.get("config") or {}, yaml.safe_load(cfg_path.read_text())
    return all(used.get(k) == v for k, v in now.items()) and res.get("judge_status") != "incomplete"


LARGE_MB = 40  # GitHub warns about files over 50 MB


def compress_large(path: Path) -> str:
    """Store a large per-answer score file gzipped (readers accept both); returns its path relative to results/."""
    if path.stat().st_size < LARGE_MB * 2**20:
        return str(path.relative_to(RESULTS))
    import gzip

    gz = path.with_name(path.name + ".gz")
    with open(path, "rb") as f, gzip.open(gz, "wb", compresslevel=9) as out:
        shutil.copyfileobj(f, out)
    path.unlink()
    return str(gz.relative_to(RESULTS))


def import_results(src: Path, dry: bool = False) -> dict:
    root = find_root(src)
    done = {p.stem for p in (root / "_checkpoints").glob("*.json") if not rerun_since(root, p)}
    report = {"finished_steps": sorted(done), "imported": [], "not_finished": []}
    for step, outputs in STEP_OUTPUTS.items():
        present = [o for o in outputs if (root / o).exists()]
        if not present:
            continue
        if step in done and not result_is_current(root, step):  # a marker from before the config changed
            done.discard(step)
        if step not in done:  # switched off or unfinished: report it only if its files differ from ours
            if copy_path_differs(root, present):
                report["not_finished"].append(step)
            continue
        for o in present:
            report["imported"] += copy_path(root / o, RESULTS / o, dry)
    report["finished_steps"] = sorted(done)
    if not dry:
        report["imported"] = [compress_large(RESULTS / f) if f.endswith("scored.jsonl") else f
                              for f in report["imported"]]
    if "latency" in done:  # measurements from an unfinished or failed latency step are left out
        report["imported"] += merge_latency(root / "latency.json", RESULTS / "latency.json", dry)
    for log in (root / "logs").glob("kaggle_*.log"):
        report["imported"] += copy_path(log, RESULTS / "logs" / log.name, dry)
    return report


def render(py: str = sys.executable) -> None:
    for exp in EXPERIMENTS:
        if (RESULTS / f"{exp}.json").exists():
            subprocess.run([py, "-m", "scripts.run_eval", "--tables-only", "--config",
                            f"configs/experiments/{exp}.yaml"], cwd=PROJECT_ROOT, check=True)
    subprocess.run([py, "-m", "scripts.paper_numbers"], cwd=PROJECT_ROOT, check=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="mhrag_results.zip or an unzipped mhrag_results folder")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        src = args.source.expanduser()
        if src.suffix == ".zip":
            zipfile.ZipFile(src).extractall(tmp)
            src = Path(tmp)
        report = import_results(src, args.dry_run)
    print("finished on Kaggle:", ", ".join(report["finished_steps"]) or "none")
    if report["not_finished"]:
        print("not finished, so left out (the next session continues them):", ", ".join(report["not_finished"]))
    print(f"{'would import' if args.dry_run else 'imported'} {len(report['imported'])} new or changed file(s):")
    for f in report["imported"]:
        print("  ", f)
    if report["imported"] and not args.dry_run:
        render()
        print("\nTables and paper/numbers.tex regenerated. Review with `git status` / `git diff`, then commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
