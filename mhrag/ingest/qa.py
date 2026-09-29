"""Loaders for the question/answer style sources: the FAQ CSV, KB.json / mentalhealth.json intents and the
CounselChat context/response pairs."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import re
from dataclasses import dataclass

import pandas as pd

from mhrag.ingest.clean import fix_mojibake
from mhrag.ingest.sources import RawData
from mhrag.types import Document, Section

FAQ_FILE = "CSV Files/Mental_Health_FAQ.csv"
FAQ_DUP_FILE = "CSV Files/mentalhealth.csv"
KB_FILE = "JSON Files/KB.json"
MH_JSON_FILE = "JSON Files/mentalhealth.json"
COUNSEL_FILE = "CSV Files/context_response_train.csv"
COUNSEL_DUP_FILE = "CSV Files/train.csv"

FAQ_URL = "https://www.kaggle.com/datasets/narendrageek/mental-health-faq-for-chatbot"
COUNSEL_URL = "https://huggingface.co/datasets/Amod/mental_health_counseling_conversations"

# Known data errors found during the audit (CHANGELOG B39).
BROKEN_INTENTS = {"fact-10": "answer duplicates fact-9 (prevalence) instead of answering 'what causes mental illness'"}


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", fix_mojibake(str(s))).strip()


def load_faq(rd: RawData) -> list[Document]:
    df = pd.read_csv(io.BytesIO(rd.read(FAQ_FILE)))
    docs = []
    for _, row in df.iterrows():
        q, a = _norm(row["Questions"]), fix_mojibake(str(row["Answers"])).strip()
        docs.append(
            Document(
                doc_id=f"faq:{row['Question_ID']}",
                title="Mental Health FAQ",
                source_file=FAQ_FILE,
                source_type="faq",
                sections=[Section(heading=q, text=a)],
                url=FAQ_URL,
                url_verified=False,
                meta={"question": q, "answer": a, "question_id": int(row["Question_ID"])},
            )
        )
    return docs


def faq_duplicate_stats(rd: RawData) -> dict:
    f = pd.read_csv(io.BytesIO(rd.read(FAQ_FILE)))
    if not rd.exists(FAQ_DUP_FILE):
        return {}
    m = pd.read_csv(io.BytesIO(rd.read(FAQ_DUP_FILE)))
    j = f.merge(m, on="Question_ID")
    return {
        "faq_rows": len(f),
        "mentalhealth_csv_rows": len(m),
        "mentalhealth_csv_rows_in_faq": len(j),
        "identical_answers": int((j.Answers_x.str.strip() == j.Answers_y.str.strip()).sum()),
        "action": "mentalhealth.csv dropped (duplicate of Mental_Health_FAQ.csv)",
    }


def _responses(intent: dict) -> list[str]:
    """KB.json uses 'responses' (list) except one intent that uses 'response' holding a *stringified* list."""
    r = intent.get("responses", intent.get("response", []))
    if isinstance(r, str):
        try:
            parsed = ast.literal_eval(r)
            r = parsed if isinstance(parsed, list) else [r]
        except (ValueError, SyntaxError):
            r = [r]
    return [fix_mojibake(str(x)).strip() for x in r if str(x).strip()]


@dataclass
class Intent:
    source_file: str
    tag: str
    patterns: list[str]
    responses: list[str]


def load_intents(rd: RawData, name: str) -> list[Intent]:
    data = json.loads(rd.read(name))
    items = data["intents"] if isinstance(data, dict) else data
    return [
        Intent(
            source_file=name,
            tag=str(it.get("tag", "")),
            patterns=[fix_mojibake(p).strip() for p in it.get("patterns", []) if p and p.strip()],
            responses=_responses(it),
        )
        for it in items
    ]


def load_kb_facts(rd: RawData) -> tuple[list[Document], dict]:
    """Only `fact-*` intents are knowledge; chit-chat intents (greetings, the 'Pandora' persona, a hard-coded
    helpline) are excluded (CHANGELOG B41)."""
    intents = load_intents(rd, KB_FILE)
    docs, dropped = [], {}
    for it in intents:
        if not it.tag.startswith("fact-"):
            continue
        if it.tag in BROKEN_INTENTS:
            dropped[it.tag] = BROKEN_INTENTS[it.tag]
            continue
        text = "\n".join(dict.fromkeys(it.responses))  # de-dup identical responses, keep order
        docs.append(
            Document(
                doc_id=f"kb:{it.tag}",
                title="Mental health facts (KB.json)",
                source_file=KB_FILE,
                source_type="kb_fact",
                sections=[Section(heading=it.patterns[0] if it.patterns else it.tag, text=text)],
                url=None,
                meta={"patterns": it.patterns, "tag": it.tag},
            )
        )
    stats = {
        "kb_intents": len(intents),
        "kb_fact_intents": sum(it.tag.startswith("fact-") for it in intents),
        "kb_fact_single_pattern": sum(it.tag.startswith("fact-") and len(it.patterns) == 1 for it in intents),
        "kb_non_fact_intents_excluded": sum(not it.tag.startswith("fact-") for it in intents),
        "dropped_intents": dropped,
    }
    return docs, stats


def counselling_split(context: str, test_fraction: float = 0.2) -> str:
    """Deterministic train/test split by hashing the context text (so Counsel-Gen test contexts never
    enter the index even when the counselling source is enabled)."""
    h = int(hashlib.sha1(_norm(context).encode()).hexdigest(), 16) % 1000
    return "test" if h < test_fraction * 1000 else "train"


def load_counselling_pairs(rd: RawData) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(rd.read(COUNSEL_FILE))).dropna()
    df["Context"] = df["Context"].map(_norm)
    df["Response"] = df["Response"].map(lambda s: str(s).strip())
    df = df.drop_duplicates()
    df["split"] = df["Context"].map(counselling_split)
    return df


def load_counselling(rd: RawData) -> list[Document]:
    df = load_counselling_pairs(rd)
    docs = []
    for ctx, g in df[df.split == "train"].groupby("Context", sort=True):
        resp = g["Response"].tolist()[0]  # one response per context keeps chunks focused
        docs.append(
            Document(
                doc_id=f"counsel:{hashlib.sha1(ctx.encode()).hexdigest()[:10]}",
                title="Counselling Q&A (CounselChat)",
                source_file=COUNSEL_FILE,
                source_type="counselling",
                sections=[Section(heading=ctx[:120], text=f"Question: {ctx}\nCounsellor: {resp}")],
                url=COUNSEL_URL,
            )
        )
    return docs


def counselling_duplicate_stats(rd: RawData) -> dict:
    if not rd.exists(COUNSEL_DUP_FILE):
        return {}
    a = pd.read_csv(io.BytesIO(rd.read(COUNSEL_FILE)))
    b = pd.read_csv(io.BytesIO(rd.read(COUNSEL_DUP_FILE)))
    return {
        "rows": len(a),
        "unique_contexts": int(a["Context"].nunique()),
        "train_csv_identical_rows": bool(a.equals(b)),
        "action": "train.csv dropped (identical rows to context_response_train.csv)",
    }
