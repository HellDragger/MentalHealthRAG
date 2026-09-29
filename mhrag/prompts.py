"""Prompts. The grounded system prompt is fixed and versioned so results are reproducible (reported in the
paper appendix)."""

from __future__ import annotations

import re
from collections.abc import Callable

PROMPT_VERSION = "grounded-v2"

SYSTEM_PROMPT = """You are a supportive mental-health information assistant. You give general psychoeducation, not therapy, diagnosis or medical advice.

Rules:
1. Answer using ONLY the numbered sources in the context. Cite them inline like [1] or [2][3] after the sentences they support.
2. If the sources do not contain the answer, say so plainly (for example: "I don't have information about that in my sources.") and suggest where the person could find help. Do not make up facts, statistics, services or phone numbers.
3. Be warm, non-judgemental and concise: at most about 200 words and at most 6 bullet points. For broad topics give the key points and offer to go into more detail. Use plain language. Never repeat the same sentence twice.
4. Never diagnose the user, never recommend starting, stopping or changing medication or doses, and never give details about methods of self-harm or suicide.
5. Encourage professional support (a doctor, therapist or helpline) when it is relevant.
6. Sources describe services in the UK (Mind) and North America (FAQ); if you mention a service, say which country it is in.
7. If the user seems to be in danger, tell them to contact emergency services or a crisis helpline immediately."""

NO_CONTEXT_SYSTEM_PROMPT = """You are a supportive mental-health information assistant. You give general psychoeducation, not therapy, diagnosis or medical advice. Be warm, concise (usually under 200 words) and accurate. If you are not sure, say so. Never diagnose, never recommend medication changes, and never give details about self-harm or suicide methods. Encourage professional support when relevant."""

# The v1 prompt, kept verbatim for the "naive RAG" baseline (no system prompt, no chat roles in v1; here it is
# sent as a single user message so every backend can run it).
V1_TEMPLATE = """
**Context:**

{context}

**Question:**
{question}

**Answer:**
"""


def source_label(chunk) -> str:
    sec = f" — {chunk.section}" if chunk.section and chunk.section != chunk.title else ""
    return f"{chunk.title}{sec}"


def build_context(hits, budget_tokens: int, count: Callable[[str], int]) -> tuple[str, list]:
    """Numbered context blocks, trimmed to a token budget (whole chunks only; at least one chunk)."""
    blocks, used, total = [], [], 0
    for h in hits:
        block = f"[{len(used) + 1}] {source_label(h.chunk)}\n{h.chunk.text.strip()}"
        n = count(block)
        if used and total + n > budget_tokens:
            break
        blocks.append(block)
        used.append(h)
        total += n
    return "\n\n".join(blocks), used


def grounded_messages(question: str, context: str, history: list[dict]) -> list[dict]:
    user = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the context above, with [n] citations."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, *history, {"role": "user", "content": user}]


def no_context_messages(question: str, history: list[dict]) -> list[dict]:
    return [{"role": "system", "content": NO_CONTEXT_SYSTEM_PROMPT}, *history, {"role": "user", "content": question}]


def v1_messages(question: str, hits) -> list[dict]:
    context = " ".join(h.chunk.text for h in hits)  # v1 joined chunks with a bare space
    return [{"role": "user", "content": V1_TEMPLATE.format(context=context, question=question)}]


TRUNCATION_NOTE = "\n\n_(I stopped here to keep the answer short. Ask me to continue for more.)_"


def trim_to_boundary(text: str) -> str:
    """Cut a length-truncated answer back to the last complete bullet or sentence."""
    text = text.rstrip()
    lines = text.split("\n")
    if len(lines) > 1 and not re.search(r"[.!?)\]:]\s*$", lines[-1]):
        lines = lines[:-1]  # drop an unfinished last bullet / line
        text = "\n".join(lines).rstrip()
    m = list(re.finditer(r"[.!?](?:\s*\[\d{1,2}\])*(?=\s|$)", text))
    if m and m[-1].end() < len(text) and not text.rstrip().endswith((".", "!", "?", ")", "]")):
        text = text[: m[-1].end()]
    return text.rstrip()
