"""Trained risk classifiers (produced by scripts/train_risk_classifier.py)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from mhrag.safety.normalize import normalize_like_training

log = logging.getLogger(__name__)


class RiskClassifier:
    name = "base"
    threshold: float = 0.5  # recall-oriented threshold chosen on the validation split
    threshold_high_precision: float = 0.9  # precision-oriented threshold (blocking crisis protocol)

    def predict_proba(self, texts: list[str]) -> list[float]:  # pragma: no cover
        raise NotImplementedError

    def score(self, text: str) -> float:
        return float(self.predict_proba([text])[0])


class TfidfLRClassifier(RiskClassifier):
    name = "tfidf_lr"

    def __init__(self, directory: Path):
        import warnings

        import joblib
        import sklearn

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.pipeline = joblib.load(directory / "tfidf_lr.joblib")
        saved = next((getattr(w.message, "original_sklearn_version", None) for w in caught
                      if type(w.message).__name__ == "InconsistentVersionWarning"), None)
        if saved and saved != sklearn.__version__:
            # A pickled model is only reliable with the scikit-learn version that saved it (e.g. 1.9 -> 1.6 fails at
            # predict time with "'LogisticRegression' object has no attribute 'multi_class'").
            raise RuntimeError(
                f"{directory / 'tfidf_lr.joblib'} was saved with scikit-learn {saved}, but {sklearn.__version__} is "
                f"installed. Install scikit-learn=={saved} (pip install -e . does this) or retrain with "
                "python -m scripts.train_risk_classifier --skip-transformer."
            )
        meta = json.loads((directory / "tfidf_lr.meta.json").read_text())
        self.threshold = float(meta["threshold"])
        self.threshold_high_precision = float(meta.get("threshold_high_precision", max(self.threshold, 0.9)))
        self.meta = meta

    def predict_proba(self, texts):
        return self.pipeline.predict_proba([normalize_like_training(t) for t in texts])[:, 1].tolist()


class TransformerClassifier(RiskClassifier):
    name = "transformer"

    def __init__(self, directory: Path, device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        d = directory / "transformer"
        self.tok = AutoTokenizer.from_pretrained(d)
        self.model = AutoModelForSequenceClassification.from_pretrained(d).eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        meta = json.loads((d / "meta.json").read_text())
        self.threshold = float(meta["threshold"])
        self.threshold_high_precision = float(meta.get("threshold_high_precision", max(self.threshold, 0.9)))
        self.meta = meta

    def predict_proba(self, texts):
        import torch

        out = []
        for i in range(0, len(texts), 64):
            batch = [normalize_like_training(t) for t in texts[i : i + 64]]
            enc = self.tok(batch, truncation=True, max_length=128, padding=True, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                logits = self.model(**enc).logits
            out += torch.softmax(logits, -1)[:, 1].tolist()
        return out


def load_classifier(kind: str, artifacts_dir: Path) -> RiskClassifier | None:
    directory = artifacts_dir / "risk_classifier"
    if kind == "none":
        return None
    try:
        if kind == "tfidf_lr":
            return TfidfLRClassifier(directory)
        if kind == "transformer":
            return TransformerClassifier(directory)
    except FileNotFoundError:
        log.warning(
            "Risk classifier %r not found in %s; the gate will use the lexicon only. "
            "Train it with: python -m scripts.train_risk_classifier",
            kind,
            directory,
        )
        return None
    raise ValueError(f"unknown classifier {kind!r}")
