"""Post-generation checks: no method/dose details, no diagnosis, no prescribing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_CHECKS = {
    # Numeric dosing, lethal amounts or method descriptions.
    "dosage_or_method": re.compile(
        r"\b(\d+(\.\d+)?\s?(mg|milligrams?|g|grams?|tablets|pills)\b.{0,40}\b(overdose|lethal|fatal|die|kill)|"
        r"(lethal|fatal|toxic) (dose|amount)\s+(is|of|would be)\b|"
        r"\b(take|swallow)\s+\d+\s+(pills|tablets))",
        re.I,
    ),
    # Telling the user they have a condition.
    "diagnosis": re.compile(
        r"\byou (definitely |clearly |probably |likely )?(have|are suffering from|suffer from|are diagnosed with)\s+"
        r"(clinical |major |severe )?(depression|bipolar|ptsd|ocd|schizophrenia|an? (anxiety|eating|personality) disorder|adhd|psychosis)\b",
        re.I,
    ),
    # Prescribing or advising medication changes.
    "prescribing": re.compile(
        r"\byou should (start|stop|increase|decrease|double|take)\s+(taking\s+)?(your\s+)?"
        r"(\d+\s?mg|antidepressants?|medication|sertraline|fluoxetine|lithium|benzodiazepines?|diazepam|xanax|pills)\b",
        re.I,
    ),
}

SAFE_REPLACEMENT = (
    "I'm not able to give that kind of detail. Decisions about diagnosis or medication should be made with a "
    "doctor or mental health professional who knows your situation. If you're struggling right now, please "
    "consider contacting one of the helplines listed on this page."
)
NOTICE = (
    "\n\n_Note: I can share general information, but I can't diagnose or recommend treatment changes. "
    "Please talk to a doctor or mental health professional about your situation._"
)


@dataclass
class OutputCheck:
    issues: list[str] = field(default_factory=list)
    text: str = ""
    replaced: bool = False

    @property
    def ok(self) -> bool:
        return not self.issues


def check_output(answer: str) -> OutputCheck:
    issues = [name for name, rx in _CHECKS.items() if rx.search(answer)]
    if "dosage_or_method" in issues:
        return OutputCheck(issues, SAFE_REPLACEMENT, replaced=True)
    if issues:
        return OutputCheck(issues, answer + NOTICE, replaced=False)
    return OutputCheck([], answer, False)
