"""History-aware query rewriting.

Follow-up turns such as "how is it treated?" retrieve nothing useful on their own. The heuristic rewriter
appends the most recent user turn's content words when the new message is short or pronoun-led; the `llm`
mode asks the active backend for a standalone question (used in experiments, off in the default server).
"""

from __future__ import annotations

import re

_FOLLOWUP = re.compile(
    r"^\s*(and|but|so|also|what about|how about|why|how|is it|does it|can it|what if)\b|\b(it|this|that|they|them|those|these)\b",
    re.I,
)
_STOP = set(
    ["i", "me", "my", "we", "you", "your", "it", "its", "this", "that", "these", "those", "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "have", "has", "had", "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "for", "with", "about", "as", "at", "by", "from", "what", "how", "why", "when", "where", "which", "who", "can", "could", "should", "would", "will", "just", "so", "very", "really", "please", "tell", "more"]
)

REWRITE_PROMPT = (
    "Rewrite the user's last message as a single standalone question that can be understood without the "
    "conversation. Keep it short. Output only the question.\n\nConversation:\n{history}\n\nLast message: {message}"
)


def heuristic_rewrite(message: str, history: list[dict]) -> str:
    prev_user = [h["content"] for h in history if h.get("role") == "user"]
    if not prev_user:
        return message
    if len(message.split()) > 12 and not _FOLLOWUP.search(message):
        return message
    if not _FOLLOWUP.search(message) and len(message.split()) > 6:
        return message
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]+", prev_user[-1]) if w.lower() not in _STOP]
    if not words:
        return message
    return f"{message} ({' '.join(dict.fromkeys(words))[:120]})"


def llm_rewrite(message: str, history: list[dict], backend, params) -> str:
    if not history:
        return message
    hist = "\n".join(f"{h['role']}: {h['content'][:300]}" for h in history[-4:])
    msgs = [{"role": "user", "content": REWRITE_PROMPT.format(history=hist, message=message)}]
    try:
        out = backend.generate(msgs, params.model_copy(update={"max_new_tokens": 48, "temperature": 0.0})).strip()
        return out.splitlines()[0][:300] if out else message
    except Exception:
        return message
