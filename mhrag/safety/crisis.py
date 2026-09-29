"""Crisis protocol responses and helpline lookup. Wording follows common safe-messaging guidance: acknowledge,
express care, encourage immediate human contact, give concrete options, never discuss methods."""

from __future__ import annotations

from functools import lru_cache

from mhrag.config import load_yaml


@lru_cache(maxsize=1)
def _helplines() -> dict:
    return load_yaml("helplines.yaml")


def regions() -> list[dict]:
    h = _helplines()
    return [{"code": c, "name": h["regions"][c]["name"]} for c in h["default_order"]]


def helplines(region: str | None = None) -> list[dict]:
    """Services for `region` first, then the other regions, then the generic fallback."""
    h = _helplines()
    order = list(h["default_order"])
    region = (region or order[0]).upper()
    if region in order:
        order.remove(region)
        order.insert(0, region)
    out = []
    for code in order:
        reg = h["regions"][code]
        out.append({"region": code, "name": reg["name"], "services": reg["services"]})
    return out


def format_helplines_md(region: str | None = None, only_primary: bool = False) -> str:
    lines = []
    for i, reg in enumerate(helplines(region)):
        if only_primary and i > 0 and reg["region"] != "INTL":
            continue
        items = []
        for s in reg["services"]:
            phone = s.get("phone")
            alt = f" / {s['alt_phone']}" if s.get("alt_phone") else ""
            items.append(f"{s['name']}: **{phone}{alt}**" if phone else s["name"])
        lines.append(f"- **{reg['name']}**: " + "; ".join(items))
    return "\n".join(lines)


CRISIS_TEMPLATE = """I'm really sorry you're feeling this way, and I'm glad you told me. What you're going through sounds very painful, and you deserve support from a person right now.

**If you might act on these thoughts or you're in danger, please contact emergency services or go to your nearest emergency department now.**

You can talk to someone right now, for free:
{helplines}

If you can, reach out to someone you trust (a friend, family member, teacher or doctor) and let them know how you're feeling. You don't have to go through this alone.

I'm an automated information service, not a counsellor, so I can't give you the support a trained person can. I'm still here if you'd like to keep talking."""

THIRD_PARTY_TEMPLATE = """It sounds like you're worried about someone's safety. That's a lot to carry, and it's good that you're taking it seriously.

**If they are in immediate danger, contact emergency services now and stay with them if it is safe to do so.**

Things that can help:
- Ask them directly and calmly whether they are thinking about suicide. Asking does not put the idea in their head.
- Listen without judging, and let them know you care.
- Encourage them to contact a helpline or a doctor, and offer to help them do it.
- Look after yourself too, and tell someone you trust what's happening.

Helplines (they also support people who are worried about someone else):
{helplines}"""

HARMFUL_TEMPLATE = """I can't help with that. I'm concerned about why you're asking, and I'd really like you to be safe.

If you're thinking about ending your life or hurting yourself, please talk to someone now:
{helplines}

If you're in immediate danger, contact emergency services. I'm here to talk about how you're feeling or to share information about getting support."""

ELEVATED_PREFIX = (
    "It sounds like things might be really hard right now. If you ever feel unsafe, please reach out to a helpline "
    "or someone you trust; there are numbers in the banner above.\n\n"
)


def crisis_message(label: str, region: str | None = None) -> str:
    hl = format_helplines_md(region, only_primary=True)
    if label == "third_party":
        return THIRD_PARTY_TEMPLATE.format(helplines=hl)
    if label == "harmful_request":
        return HARMFUL_TEMPLATE.format(helplines=hl)
    return CRISIS_TEMPLATE.format(helplines=hl)
