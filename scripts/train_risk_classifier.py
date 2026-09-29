"""Train and evaluate the risk-triage classifiers.

    python -m scripts.train_risk_classifier                 # TF-IDF + LR (+ transformer if torch is available)
    python -m scripts.train_risk_classifier --skip-transformer
    python -m scripts.train_risk_classifier --transformer-model mental/mental-roberta-base   # needs HF access

Training data: `mental_health.csv` (label 1 = suicidal/at-risk post, 0 = not), de-duplicated, stratified 80/10/10
split with seed 42. The decision threshold is the largest one reaching recall >= 0.95 on the validation split
(recall-favouring, as missing a crisis is costlier than a false alarm).

Zero-shot cross-dataset tests (same threshold): depression_dataset_reddit_cleaned (depression vs not),
dreaddit (stress vs not), Mental-Health-Twitter (depression users vs not). Their labels are *related but not
identical* constructs, so these numbers measure transfer, not suicide-risk accuracy.
In-the-wild probe: the 97 `suicide` intent patterns of KB.json (all crisis statements) and the distress/venting
intents (sad, stressed, worthless, depressed, anxious, sleep, scared, death), which are distress but not labelled
suicidal (a proxy for over-triggering).

Outputs: artifacts/risk_classifier/{tfidf_lr.joblib, tfidf_lr.meta.json, transformer/},
results/risk_classifier.json, paper/tables/risk_classifier*.tex
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import time

import numpy as np
import pandas as pd

from eval.metrics import binary_report, bootstrap_metric, threshold_for_precision, threshold_for_recall
from eval.tables import fmt, fmt_ci, write_table
from mhrag.config import get_settings
from mhrag.ingest import qa
from mhrag.ingest.sources import RawData
from mhrag.safety.normalize import normalize_like_training

log = logging.getLogger("train_risk")
SEED = 42
TARGET_RECALL = 0.95
TARGET_PRECISION = 0.95  # high-precision threshold that triggers the blocking crisis protocol
DISTRESS_INTENTS = ("sad", "stressed", "worthless", "depressed", "anxious", "sleep", "scared", "death")


def load_csv(rd: RawData, name: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(rd.read(f"CSV Files/{name}")))


def dedup(df: pd.DataFrame, text="text", label="label") -> tuple[pd.DataFrame, dict]:
    df = df.dropna(subset=[text]).copy()
    df[text] = df[text].astype(str)
    df = df[df[text].str.strip().str.len() > 0]
    n0 = len(df)
    conflict = df.groupby(text)[label].nunique()
    bad = set(conflict[conflict > 1].index)
    df = df[~df[text].isin(bad)].drop_duplicates(subset=[text])
    return df, {"rows": n0, "conflicting_label_texts_dropped": len(bad), "after_dedup": len(df)}


def split(df: pd.DataFrame, label="label"):
    from sklearn.model_selection import train_test_split

    train, rest = train_test_split(df, test_size=0.2, stratify=df[label], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest[label], random_state=SEED)
    return train, val, test


def fit_tfidf(train, val):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline

    best = None
    for c in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True)),
            ("lr", LogisticRegression(C=c, class_weight="balanced", max_iter=3000, solver="liblinear")),
        ])
        pipe.fit(train.x, train.label)
        auc = roc_auc_score(val.label, pipe.predict_proba(val.x)[:, 1])
        log.info("tfidf C=%.1f val AUROC=%.4f", c, auc)
        if best is None or auc > best[0]:
            best = (auc, c, pipe)
    return best[2], {"C": best[1], "val_auroc": best[0], "C_grid": [0.5, 1, 2, 4, 8, 16, 32]}


def fit_transformer(train, val, model_name: str, out_dir, epochs: int = 2, max_len: int = 128, bs: int = 32,
                    lr: float = 3e-5):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    xs, ys = train.x.tolist(), train.label.to_numpy()
    steps = epochs * ((len(xs) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.06 * steps)) * max(0.0, (steps - s) / steps))
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    step = 0
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(xs))
        for i in range(0, len(xs), bs):
            idx = order[i : i + bs]
            enc = tok([xs[j] for j in idx], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(device)
            out = model(**enc, labels=torch.tensor(ys[idx], device=device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 100 == 0:
                log.info("epoch %d step %d/%d loss %.4f (%.0fs)", ep, step, steps, out.loss.item(), time.time() - t0)
    model.eval()

    def predict(texts):
        res = []
        with torch.inference_mode():
            for i in range(0, len(texts), 128):
                enc = tok(texts[i : i + 128], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(device)
                res += torch.softmax(model(**enc).logits, -1)[:, 1].float().cpu().tolist()
        return np.asarray(res)

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return predict, {"model": model_name, "epochs": epochs, "max_len": max_len, "batch_size": bs, "lr": lr,
                     "device": device, "train_seconds": time.time() - t0}


def load_transformer(path, max_len: int = 128):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
    meta = json.loads((path / "meta.json").read_text()) if (path / "meta.json").exists() else {}

    def predict(texts):
        res = []
        with torch.inference_mode():
            for i in range(0, len(texts), 128):
                enc = tok(texts[i : i + 128], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(device)
                res += torch.softmax(model(**enc).logits, -1)[:, 1].float().cpu().tolist()
        return np.asarray(res)

    keys = ("model", "epochs", "max_len", "batch_size", "lr", "device", "train_seconds")
    info = {k: meta[k] for k in keys if k in meta}
    prev = get_settings().results_dir / "risk_classifier.json"  # training info from the run that produced the checkpoint
    if prev.exists():
        old = json.loads(prev.read_text()).get("models", {}).get("transformer", {})
        info = {**{k: old[k] for k in keys if k in old}, **info}
    return predict, {**info, "reused_checkpoint": True}


def evaluate(name: str, predict, val, test, externals: dict) -> dict:
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    val_scores = predict(val.x.tolist())
    tau = threshold_for_recall(val.label, val_scores, TARGET_RECALL)
    tau90 = threshold_for_recall(val.label, val_scores, 0.90)
    tau99 = threshold_for_recall(val.label, val_scores, 0.99)
    # The blocking tier must never be looser than the elevated tier.
    tau_hp = max(tau, threshold_for_precision(val.label, val_scores, TARGET_PRECISION))
    test_scores = predict(test.x.tolist())
    y = test.label.to_numpy()
    rep = {
        "threshold": tau,
        "threshold_rule": f"largest threshold with validation recall >= {TARGET_RECALL}",
        "threshold_high_precision": tau_hp,
        "threshold_high_precision_rule": f"max(tau, smallest threshold with validation precision >= {TARGET_PRECISION})",
        "val": binary_report(val.label, val_scores, tau),
        "test_at_high_precision": binary_report(test.label.to_numpy(), predict(test.x.tolist()), tau_hp),
        "test_at_0.5": binary_report(y, test_scores, 0.5),
        "test_at_tau": binary_report(y, test_scores, tau),
        "test_at_recall_targets": {
            "0.90": binary_report(y, test_scores, tau90),
            "0.95": binary_report(y, test_scores, tau),
            "0.99": binary_report(y, test_scores, tau99),
        },
        "test_ci95": {
            "auroc": bootstrap_metric(y, test_scores, roc_auc_score),
            "recall": bootstrap_metric(y, test_scores, lambda a, s: recall_score(a, s >= tau)),
            "precision": bootstrap_metric(y, test_scores, lambda a, s: precision_score(a, s >= tau, zero_division=0)),
            "f1": bootstrap_metric(y, test_scores, lambda a, s: f1_score(a, s >= tau)),
        },
        "external": {},
    }
    for ext, (texts, labels) in externals.items():
        s = predict(texts)
        rep["external"][ext] = binary_report(labels, s, tau) if labels is not None else {
            "n": len(texts), "flag_rate_at_tau": float((s >= tau).mean()), "threshold": tau}
    log.info("%s: test AUROC %.4f recall@tau %.4f precision@tau %.4f", name, rep["test_at_tau"]["auroc"],
             rep["test_at_tau"]["recall"], rep["test_at_tau"]["precision"])
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-transformer", action="store_true")
    ap.add_argument("--transformer-model", default="distilbert/distilroberta-base")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--reuse-transformer", action="store_true",
                    help="evaluate the already fine-tuned model in artifacts/risk_classifier/transformer (no retraining)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    s = get_settings()
    rd = RawData(s.resolve(s.paths.raw_data))
    art = s.artifacts_dir / "risk_classifier"
    art.mkdir(parents=True, exist_ok=True)

    df, dstats = dedup(load_csv(rd, "mental_health.csv"))
    df["x"] = df.text.map(normalize_like_training)
    train, val, test = split(df)
    split_info = {
        "source": "CSV Files/mental_health.csv", "seed": SEED, "dedup": dstats,
        "sizes": {k: {"n": len(v), "positives": int(v.label.sum())} for k, v in (("train", train), ("val", val), ("test", test))},
        "test_ids_sha1": hashlib.sha1(",".join(map(str, sorted(test.index))).encode()).hexdigest(),
    }

    # External / probe sets (normalised with the same function)
    ext: dict = {}
    dep, dep_st = dedup(load_csv(rd, "depression_dataset_reddit_cleaned.csv"), "clean_text", "is_depression")
    ext["reddit_depression"] = (dep.clean_text.map(normalize_like_training).tolist(), dep.is_depression.to_numpy())
    dr = pd.concat([load_csv(rd, "dreaddit-train.csv"), load_csv(rd, "dreaddit-test.csv")], ignore_index=True)
    dr, dr_st = dedup(dr[["text", "label"]])
    ext["dreaddit_stress"] = (dr.text.map(normalize_like_training).tolist(), dr.label.to_numpy())
    tw, tw_st = dedup(load_csv(rd, "Mental-Health-Twitter.csv").rename(columns={"post_text": "text"})[["text", "label"]])
    ext["twitter_depression"] = (tw.text.map(normalize_like_training).tolist(), tw.label.to_numpy())
    intents = qa.load_intents(rd, qa.KB_FILE)
    crisis = [p for it in intents if it.tag == "suicide" for p in it.patterns]
    distress = [p for it in intents if it.tag in DISTRESS_INTENTS for p in it.patterns]
    ext["kb_suicide_intent_patterns"] = ([normalize_like_training(t) for t in crisis], np.ones(len(crisis), int))
    ext["kb_distress_intent_patterns"] = ([normalize_like_training(t) for t in distress], None)
    ext_info = {"reddit_depression": dep_st, "dreaddit_stress": dr_st, "twitter_depression": tw_st,
                "kb_suicide_intent_patterns": {"n": len(crisis)},
                "kb_distress_intent_patterns": {"n": len(distress), "intents": list(DISTRESS_INTENTS)}}

    results = {"split": split_info, "external_sets": ext_info, "models": {}}

    pipe, tf_info = fit_tfidf(train, val)
    rep = evaluate("tfidf_lr", lambda t: pipe.predict_proba(t)[:, 1], val, test, ext)
    results["models"]["tfidf_lr"] = {**tf_info, **rep}
    import joblib

    joblib.dump(pipe, art / "tfidf_lr.joblib", compress=3)
    (art / "tfidf_lr.meta.json").write_text(json.dumps({
        "threshold": rep["threshold"], "threshold_rule": rep["threshold_rule"],
        "threshold_high_precision": rep["threshold_high_precision"], "C": tf_info["C"],
        "trained_on": "mental_health.csv", "normalizer": "mhrag.safety.normalize.normalize_like_training",
        "seed": SEED}, indent=2))

    if not args.skip_transformer:
        try:
            if args.reuse_transformer and (art / "transformer" / "config.json").exists():
                predict, tr_info = load_transformer(art / "transformer")
            else:
                predict, tr_info = fit_transformer(train, val, args.transformer_model, art / "transformer", args.epochs)
            rep = evaluate("transformer", predict, val, test, ext)
            results["models"]["transformer"] = {**tr_info, **rep}
            (art / "transformer" / "meta.json").write_text(json.dumps(
                {"threshold": rep["threshold"], "threshold_high_precision": rep["threshold_high_precision"],
                 "base_model": args.transformer_model, "seed": SEED, **tr_info}, indent=2))
        except Exception as e:  # e.g. gated model without access
            log.error("transformer training failed: %s", e)
            results["models"]["transformer"] = {"status": f"failed: {type(e).__name__}: {e}"}

    s.results_dir.mkdir(parents=True, exist_ok=True)
    (s.results_dir / "risk_classifier.json").write_text(json.dumps(results, indent=2))
    write_tables(results)
    print(json.dumps({k: {m: v.get("test_at_tau", v) for m, v in [(k, results["models"][k])]} for k in results["models"]}, indent=1)[:3000])


def write_tables(results: dict):
    rows = []
    names = {"tfidf_lr": "TF-IDF + LR", "transformer": "DistilRoBERTa (fine-tuned)"}
    for k, m in results["models"].items():
        if "test_at_tau" not in m:
            continue
        ci = m["test_ci95"]
        rows.append([names.get(k, k), fmt(m["threshold"]), fmt_ci(*ci["precision"]), fmt_ci(*ci["recall"]),
                     fmt_ci(*ci["f1"]), fmt_ci(*ci["auroc"]), fmt(m["test_at_tau"]["auprc"])])
    sz = results["split"]["sizes"]["test"]
    write_table("risk_classifier", ["Model", "$\\tau$", "Precision", "Recall", "F1", "AUROC", "AUPRC"], rows,
                caption=f"Risk classifier on the held-out test split of \\texttt{{mental\\_health.csv}} "
                        f"(n={sz['n']}, {sz['positives']} positive). $\\tau$ is the largest threshold with validation "
                        "recall $\\geq 0.95$. 95\\% bootstrap CIs in brackets.",
                label="tab:risk-classifier", source="results/risk_classifier.json")
    ext_names = {"reddit_depression": "Reddit depression", "dreaddit_stress": "Dreaddit (stress)",
                 "twitter_depression": "Twitter depression", "kb_suicide_intent_patterns": "KB.json suicide patterns"}
    rows = []
    for ext, label in ext_names.items():
        row = [label]
        for k in results["models"]:
            e = results["models"][k].get("external", {}).get(ext)
            if not e:
                row += ["--", "--", "--"]
                continue
            row += [fmt(e.get("precision")) if ext != "kb_suicide_intent_patterns" else "--", fmt(e.get("recall")),
                    fmt(e.get("auroc"))]
        rows.append(row)
    fp = []
    for k in results["models"]:
        e = results["models"][k].get("external", {}).get("kb_distress_intent_patterns")
        fp.append(f"{names.get(k, k)}: {e['flag_rate_at_tau']:.3f}" if e else "")
    header = ["Dataset"]
    for k in results["models"]:
        header += [f"{names.get(k, k)} P", "R", "AUROC"]
    write_table("risk_classifier_transfer", header, rows,
                caption="Zero-shot transfer of the risk classifiers (threshold fixed at $\\tau$). Labels of the external "
                        "sets are related constructs (depression, stress), not suicide risk.",
                label="tab:risk-transfer", source="results/risk_classifier.json",
                note="Flag rate on KB.json distress/venting patterns (not labelled suicidal): " + "; ".join(fp))


if __name__ == "__main__":
    main()
