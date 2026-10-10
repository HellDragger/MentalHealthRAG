"""Cross-check the phrase-list abstention metric with an LLM judge on the out-of-scope answers.

    python -m scripts.judge_abstention            # needs GROQ_API_KEY; -> results/abstention_check.json

The phrase list (eval/judges.py) includes the sentence the full pipeline's system prompt suggests, so it may favour the
full pipeline. Here an LLM judge, which never sees the system prompt, labels each saved out-of-scope answer as
"declined" or "answered", and the script reports per setting both rates, their agreement (Cohen's kappa) and the
full-versus-naive difference under each measure. Verdicts are cached, so an interrupted run (API quota) continues where
it stopped; exit code 75 means "quota used up, run again later".
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys

import numpy as np

from eval.judges import _parse_json
from eval.metrics import paired_bootstrap_p
from eval.tables import write_table
from mhrag.config import get_settings

R = get_settings().results_dir
PROMPT_VERSION = "abstain-v1"
PROMPT = """A mental-health information chatbot received a question that is outside its scope (off-topic, unanswerable from mental-health leaflets, or adversarial). Classify its RESPONSE.

- "declined": the response does not answer the question. It says it lacks the information, declines, or only redirects the user elsewhere.
- "answered": the response gives substantive content that answers the question, even if it adds a caveat or a referral.

QUESTION: {question}
RESPONSE: {answer}

Return ONLY a JSON object like {{"label": "declined"}} or {{"label": "answered"}}."""
# experiment:model:profiles checked by default (about 250 judge calls)
DEFAULT = ["local:qwen2.5-1.5b-gguf:no_rag,naive,full", "ablation:qwen2.5-1.5b-gguf:abl_naive_full,abl_full_naive"]


def load(exp: str) -> list[dict]:
    d = R / "generation" / exp
    for name, op in (("scored.jsonl", open), ("scored.jsonl.gz", gzip.open)):
        if (d / name).exists():
            with op(d / name, "rt", encoding="utf-8") as f:
                return [json.loads(line) for line in f]
    return []


def kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=DEFAULT, help="experiment:model:profile1,profile2 ...")
    ap.add_argument("--judge", default="gpt-oss-120b-groq")
    args = ap.parse_args(argv)
    from mhrag.llm.base import BackendError, GenerationParams
    from mhrag.llm.registry import ModelManager, availability, load_catalog

    catalog = load_catalog()
    ok, why = availability(catalog[args.judge])
    if not ok:
        print(f"judge {args.judge} unavailable: {why}")
        return 1
    backend = ModelManager(catalog, [args.judge], args.judge).get(args.judge)
    params = GenerationParams(max_new_tokens=1024, temperature=0.0)
    cache_path = R / "generation" / "abstention_judge_cache.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            c = json.loads(line)
            cache[c["key"]] = c["label"]

    out: dict = {"judge": args.judge, "prompt_version": PROMPT_VERSION, "sets": {}}
    complete = True
    for spec in args.sets:
        exp, model, profiles = spec.split(":")
        rows = [r for r in load(exp) if r["model"] == model and r["dataset"] == "oos_questions" and r.get("generated")]
        if not rows:
            print(f"{spec}: no saved out-of-scope answers yet")
            complete = False
            continue
        res: dict = {}
        for prof in profiles.split(","):
            items = {r["id"]: r for r in rows if r["profile"] == prof}
            labels = {}
            for i, r in sorted(items.items()):
                key = hashlib.sha1(json.dumps([args.judge, PROMPT_VERSION, r["query"], r["answer"]]).encode()).hexdigest()
                if key not in cache:
                    try:
                        text = backend.generate([{"role": "user", "content": PROMPT.format(
                            question=r["query"], answer=(r["answer"] or "")[:3000])}], params)
                    except BackendError as e:
                        print(f"judge stopped ({e}); cached verdicts are kept, run again later")
                        return 75 if "429" in str(e) else 1
                    lab = (_parse_json(text) or {}).get("label")
                    cache[key] = lab if lab in ("declined", "answered") else None
                    with open(cache_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"key": key, "label": cache[key]}) + "\n")
                if cache[key] is not None:
                    labels[i] = int(cache[key] == "declined")
            ids = sorted(labels)
            rx = [int(bool(items[i]["metrics"]["abstained"])) for i in ids]
            jd = [labels[i] for i in ids]
            res[prof] = {"n": len(ids), "unparsed": len(items) - len(ids), "regex_rate": float(np.mean(rx)),
                         "judge_rate": float(np.mean(jd)), "agreement": float(np.mean(np.equal(rx, jd))),
                         "cohen_kappa": kappa(rx, jd),
                         "regex_only": int(sum(a and not b for a, b in zip(rx, jd, strict=True))),
                         "judge_only": int(sum(b and not a for a, b in zip(rx, jd, strict=True))),
                         "_judge": dict(zip(ids, jd, strict=True))}
        # does the full-vs-naive gain survive the judge? (paired on the same questions)
        for a, b in (("full", "naive"), ("full", "no_rag"), ("abl_naive_full", "naive"), ("abl_full_naive", "naive")):
            ja, jb = (res.get(a) or {}).get("_judge"), (res.get(b) or {}).get("_judge")
            if ja and jb:
                ids = [i for i in ja if i in jb]
                x, y = [ja[i] for i in ids], [jb[i] for i in ids]
                res[f"{a}>{b}"] = {"judge_delta": float(np.mean(x) - np.mean(y)), "p_bootstrap": paired_bootstrap_p(x, y),
                                   "n": len(ids)}
        for v in res.values():
            v.pop("_judge", None)
        out["sets"][spec] = res
    out["complete"] = complete
    (R / "abstention_check.json").write_text(json.dumps(out, indent=2))

    names = {"no_rag": "No RAG", "naive": "Naive RAG", "full": "Full pipeline",
             "abl_naive_full": "Naive retrieval + full prompt", "abl_full_naive": "Full retrieval + naive prompt"}
    trows = []
    for res in out["sets"].values():
        for prof, v in res.items():
            if ">" in prof:
                continue
            trows.append([names.get(prof, prof), str(v["n"]), f"{v['regex_rate']:.2f}", f"{v['judge_rate']:.2f}",
                          f"{v['agreement']:.2f}", f"{v['cohen_kappa']:.2f}" if v["cohen_kappa"] == v["cohen_kappa"] else "--"])
    if trows:
        write_table("abstention_check", ["Setting", "$n$", "Declined (phrases)", "Declined (judge)", "Agreement",
                                         "Cohen $\\kappa$"], trows,
                    "Out-of-scope questions answered by Qwen2.5-1.5B (GGUF): share declined according to the phrase "
                    f"list and according to an LLM judge ({args.judge.replace('-groq', '')}) that does not see the "
                    "system prompt.", "tab:abstention-check", align="lrrrrr", source="results/abstention_check.json")
    print(json.dumps({s: {p: {k: v[k] for k in ("regex_rate", "judge_rate", "cohen_kappa")} if "regex_rate" in v else v
                          for p, v in r.items()} for s, r in out["sets"].items()}, indent=1))
    return 0 if complete else 75


if __name__ == "__main__":
    sys.exit(main())
