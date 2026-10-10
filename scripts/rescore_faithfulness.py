"""Re-score NLI faithfulness of saved answers with one premise definition for every setting and experiment.

    python -m scripts.rescore_faithfulness                 # local and gpu experiments, then re-aggregate them

Faithfulness is the share of answer sentences entailed (NLI) by at least one retrieved passage. The first runs used
different premises per setting: naive RAG scored against its two chunks joined into one premise, and the full pipeline
against numbered blocks that include a title line. The component ablation scored each retrieved passage's text as its
own premise. Here every naive and full-pipeline answer is re-scored the ablation's way, so the settings and the
experiments are comparable (identical answers then get identical scores).

Premises come from the saved generation checkpoints (results/generation/<exp>/<model>__<profile>__<dataset>.jsonl):
full-pipeline blocks lose their "[n] title" line; naive contexts (two chunks joined by a space) are split back into
their chunks using the ablation's saved passages or, for Counsel-Gen, the chunk texts of the naive index.
"""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
from collections import Counter

from eval.judges import NLIJudge
from mhrag.config import PROJECT_ROOT, get_settings

R = get_settings().results_dir


def latest(path) -> dict:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] not in out or out[r["id"]].get("error"):
                out[r["id"]] = r
    return out


def full_passages(context: str) -> list[str]:
    return [b.split("\n", 1)[1] if "\n" in b else "" for b in context.split("\n\n[")]


class NaiveSplitter:
    """Split 'chunk1 chunk2' back into the two chunks."""

    def __init__(self):
        self.known = {}
        abl = R / "generation" / "ablation" / "scored.jsonl"
        if abl.exists():
            with open(abl, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    if r["profile"] == "abl_naive_naive":
                        self.known[" ".join(r["passages"])] = r["passages"]
        self.chunks = []
        for variant in ("minilm-c48", "minilm-c48-nofaq"):
            p = PROJECT_ROOT / "artifacts" / "index" / variant / "chunks.jsonl"
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    self.chunks += [json.loads(line)["text"] for line in f]
        self.chunks = sorted(set(self.chunks), key=len, reverse=True)

    def __call__(self, context: str) -> tuple[list[str], str]:
        if context in self.known:
            return self.known[context], "ablation"
        for t in self.chunks:
            if context.startswith(t + " ") and context[len(t) + 1:] in self.chunks_set:
                return [t, context[len(t) + 1:]], "index"
        return [context], "unsplit"

    @property
    def chunks_set(self):
        if not hasattr(self, "_set"):
            self._set = set(self.chunks)
        return self._set


def main() -> int:
    nli = NLIJudge("cross-encoder/nli-deberta-v3-base")
    split = NaiveSplitter()
    for exp in ("local", "gpu"):
        d = R / "generation" / exp
        path = next((p for p in (d / "scored.jsonl", d / "scored.jsonl.gz") if p.exists()), None)
        if path is None:
            continue
        op = gzip.open if path.suffix == ".gz" else open
        with op(path, "rt", encoding="utf-8") as f:
            answers = [json.loads(line) for line in f]
        ckpt: dict = {}
        stats = Counter()
        for r in answers:
            if r["profile"] not in ("naive", "full") or not r.get("generated"):
                continue
            key = (r["model"], r["profile"], r["dataset"])
            if key not in ckpt:
                p = d / f"{r['model']}__{r['profile']}__{r['dataset']}.jsonl"
                ckpt[key] = latest(p) if p.exists() else {}
            src = ckpt[key].get(r["id"])
            if not src or not src.get("context"):
                stats["no_context"] += 1
                continue
            if r["profile"] == "full":
                passages, how = full_passages(src["context"]), "full"
            else:
                passages, how = split(src["context"])
            stats[how] += 1
            r["metrics"].update(nli.score(r["answer"] or "", [p for p in passages if p.strip()][:8]))
            r["metrics"]["nli_premises"] = "passages-v2"
        with op(path, "wt", encoding="utf-8") as f:
            for r in answers:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{exp}: re-scored {dict(stats)}", flush=True)
        subprocess.run([sys.executable, "-m", "scripts.run_eval", "--config",
                        f"configs/experiments/generation_{exp}.yaml", "--aggregate-only"], cwd=PROJECT_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
