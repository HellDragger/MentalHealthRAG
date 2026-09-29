"""Per-family chat-template quirks (fixes CHANGELOG B5: every model gets its own template)."""

from __future__ import annotations

import re

from mhrag.llm.base import Message

# Families whose templates reject a "system" role: merge it into the first user turn.
NO_SYSTEM_ROLE = {"gemma"}
_THINK = re.compile(r"<think>.*?</think>\s*", re.S)


def merge_system(messages: list[Message]) -> list[Message]:
    sys = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    rest = [dict(m) for m in messages if m["role"] != "system"]
    if sys and rest and rest[0]["role"] == "user":
        rest[0]["content"] = f"{sys}\n\n{rest[0]['content']}"
    elif sys:
        rest.insert(0, {"role": "user", "content": sys})
    return rest


def prepare_messages(messages: list[Message], family: str) -> list[Message]:
    if family in NO_SYSTEM_ROLE:
        return merge_system(messages)
    return [dict(m) for m in messages]


def llama2_prompt(messages: list[Message]) -> str:
    """Llama-2-chat format, used for MentaLLaMA whose repo ships no chat_template."""
    sys = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    turns = [m for m in messages if m["role"] != "system"]
    out, first_user = "", True
    for m in turns:
        if m["role"] == "user":
            content = m["content"]
            if first_user and sys:
                content = f"<<SYS>>\n{sys}\n<</SYS>>\n\n{content}"
            first_user = False
            out += f"<s>[INST] {content.strip()} [/INST]"
        else:
            out += f" {m['content'].strip()} </s>"
    return out


def template_kwargs(family: str) -> dict:
    # Qwen3 defaults to "thinking" mode; the chatbot and the benchmark use non-thinking mode.
    return {"enable_thinking": False} if family == "qwen3" else {}


def strip_reasoning(text: str) -> str:
    return _THINK.sub("", text)


def completion_prompt(messages: list[Message]) -> str:
    """Plain-text prompt for base LMs (GPT-2, GPT-J) and seq2seq models (BART, T5), which have no chat template.
    A single user message with no system prompt (the v1 set-up) is passed through unchanged."""
    if len(messages) == 1 and messages[0]["role"] == "user":
        return messages[0]["content"]
    parts = []
    for m in messages:
        if m["role"] == "system":
            parts.append(m["content"].strip() + "\n")
        elif m["role"] == "user":
            parts.append(f"User: {m['content'].strip()}")
        else:
            parts.append(f"Assistant: {m['content'].strip()}")
    return "\n".join(parts) + "\nAssistant:"


def qwen3_no_think(messages: list[Message]) -> list[Message]:
    """Qwen3 soft switch for backends where enable_thinking cannot be passed (llama.cpp, APIs)."""
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m["role"] == "user":
            m["content"] = m["content"] + " /no_think"
            break
    return out
