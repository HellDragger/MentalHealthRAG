"""Pre-retrieval risk-triage gate: lexicon (raw text) + trained classifier (normalised text).

Labels
- crisis:           first-person risk (lexicon crisis category, or classifier >= the high-precision threshold on a
                    first-person, non-informational message) -> skip generation, return the crisis protocol.
- harmful_request:  method / overdose information requests -> refuse, redirect to support (+ helplines).
- third_party:      someone else is at risk -> supportive guidance for helping them + helplines.
- elevated:         classifier >= the recall-oriented threshold (validation recall 0.95) but below the
                    high-precision one -> answer normally, prepend a gentle check-in and show helplines.
                    Together the two tiers favour recall without blocking every sad message.
- none:             normal RAG answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from mhrag.safety.classifier import RiskClassifier
from mhrag.safety.lexicon import scan

_FIRST_PERSON = re.compile(r"\b(i|i'm|im|me|my|myself|i've|ive|i'd|i'll)\b", re.I)
# Question-form, third-person informational requests ("what is depression?") are never escalated by the
# classifier alone; the classifier was trained on first-person social-media posts.
_INFO_Q = re.compile(r"^\s*(what|how|why|when|where|who|which|is|are|can|does|do|should|could|tell me|explain|define)\b", re.I)


@dataclass
class GateResult:
    label: str
    lexicon: dict = field(default_factory=dict)
    classifier_score: float | None = None
    classifier_threshold: float | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def blocks_generation(self) -> bool:
        return self.label in ("crisis", "harmful_request")

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "categories": sorted(self.lexicon),
            "classifier_score": None if self.classifier_score is None else round(self.classifier_score, 4),
            "reasons": self.reasons,
        }


GATE_VERSION = "v2"


class SafetyGate:
    def __init__(
        self,
        classifier: RiskClassifier | None = None,
        crisis_threshold: float | None = None,
        elevated_threshold: float | None = None,
        use_lexicon: bool = True,
        use_classifier: bool = True,
    ):
        self.classifier = classifier if use_classifier else None
        self.crisis_threshold = crisis_threshold if crisis_threshold is not None else (
            classifier.threshold_high_precision if classifier else 1.01
        )
        self.elevated_threshold = elevated_threshold if elevated_threshold is not None else (
            classifier.threshold if classifier else 1.01
        )
        self.use_lexicon = use_lexicon

    def assess(self, message: str, history: list[dict] | None = None) -> GateResult:
        lex = scan(message) if self.use_lexicon else None
        # A risk statement in the previous user turn keeps the conversation in crisis mode for short replies
        # ("yes", "tonight"), which otherwise carry no signal.
        prev_user = [h["content"] for h in (history or []) if h.get("role") == "user"]
        if self.use_lexicon and prev_user and len(message.split()) <= 4 and not _INFO_Q.match(message):
            prev = scan(prev_user[-1])
            if prev.crisis and not lex.crisis:
                lex.categories.setdefault("suicidal", []).append("(previous turn)")

        score = self.classifier.score(message) if self.classifier else None
        reasons: list[str] = []
        first_person = bool(_FIRST_PERSON.search(message))
        informational = bool(_INFO_Q.match(message)) and not first_person

        if lex and lex.method_request:
            reasons.append("lexicon:method_request")
            return GateResult("harmful_request", lex.categories, score, self.crisis_threshold, reasons)
        if lex and lex.crisis:
            reasons += [f"lexicon:{c}" for c in lex.categories if c != "third_party"]
            return GateResult("crisis", lex.categories, score, self.crisis_threshold, reasons)
        # Someone else at risk takes priority over a classifier-only signal (the classifier cannot tell who is at risk).
        if lex and lex.third_party:
            reasons.append("lexicon:third_party")
            return GateResult("third_party", lex.categories, score, self.crisis_threshold, reasons)
        negated = bool(lex and lex.has_negation)
        if (score is not None and score >= self.crisis_threshold and first_person and not informational
                and not negated):
            reasons.append(f"classifier>={self.crisis_threshold:.3f}")
            return GateResult("crisis", lex.categories if lex else {}, score, self.crisis_threshold, reasons)
        if score is not None and score >= self.elevated_threshold and not informational:
            reasons.append(f"classifier>={self.elevated_threshold:.2f}" + (" (negated: not blocking)" if negated else ""))
            return GateResult("elevated", lex.categories if lex else {}, score, self.crisis_threshold, reasons)
        return GateResult("none", lex.categories if lex else {}, score, self.crisis_threshold, reasons)
