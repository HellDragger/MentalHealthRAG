"""Curated high-precision risk lexicon, applied to raw (un-normalised) text.

Categories
- suicidal:        first-person suicidal ideation / intent / preparation
- self_harm:       first-person self-harm
- harm_others:     first-person intent to hurt someone else
- abuse:           the user reports being abused / in danger from someone
- third_party:     someone else is at risk ("my friend wants to kill himself")
- method_request:  requests for suicide / self-harm methods or overdose amounts (harmful information)

Design choices (documented in the paper): patterns require first-person framing so informational questions
("what are the warning signs of suicide?") are *not* treated as a crisis, and a negation guard suppresses
statements such as "I'm not suicidal". Implicit phrasings ("everyone would be better off without me") are
included because they are common in real crisis messages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_I = r"(?:\bi\b|\bi'?m\b|\bim\b|\bi am\b|\bi've\b|\bive\b|\bi have\b|\bi'll\b|\bi will\b)"
_WANT = r"(?:want(?:s|ed)?|wanna|going|gonna|plan(?:ning)?|ready|trying|decided|intend(?:ing)?|thinking (?:about|of)|thought about|about|need)"
# Strong outcomes are unambiguous; weak ones ("die") need a volitional verb, because "I feel like I'm going to
# die" is a common description of a panic attack, not suicidal intent.
_DIE_STRONG = (
    r"(?:kill(?:ing)? my ?self|end(?:ing)? (?:it all|my life|my own life|everything)|take my (?:own )?life|"
    r"commit(?:ting)? suicide|hang my ?self|disappear forever|jump (?:off|in front)|overdose\b|od\b)"
)
_DIE_WEAK = r"(?:die|be dead|not (?:wake up|be here|exist)|end it|end things)"
_WILL = r"(?:want(?:s|ed)?|wanna|wish|ready|plan(?:ning)?|decided|intend(?:ing)?|deserve)"

PATTERNS: dict[str, list[str]] = {
    "suicidal": [
        rf"{_I}[^.?!\n]{{0,40}}\b{_WANT}\b[^.?!\n]{{0,20}}\b(?:to )?{_DIE_STRONG}",
        rf"{_I}[^.?!\n]{{0,30}}\b{_WILL}\b[^.?!\n]{{0,12}}\b(?:to )?{_DIE_WEAK}\b",
        r"\b(?:just |really |honestly )?(?:want|wanna|wish) (?:to )?(?:kill my ?self|end my life|die)\b",
        r"\bi(?:'m| am|m)? (?:feeling |so |really |very )*suicidal\b",
        r"\b(?:have|having|get|getting|had) (?:suicidal|suicide) (?:thoughts|feelings|urges|ideation)\b",
        r"\bdon'?t want to (?:live|be alive|exist|be here|wake up|go on)\b",
        r"\b(?:no|not any) (?:reason|point) (?:to|in) (?:live|living|going on|being alive)\b",
        r"\b(?:every(?:one|body)|they|people|my family|the world) (?:would be|will be|is|are) better off without me\b",
        r"\bbetter off dead\b",
        r"\bwish (?:i|i'd) (?:was|were|had never been) (?:dead|born|never born)\b|\bwish i could (?:die|disappear|not wake up|sleep forever)\b",
        r"\b(?:can'?t|cannot) (?:go on|keep going|do this anymore|take (?:it|this) anymore)\b.*\b(?:die|end|over|life)\b",
        r"\blife (?:is|isn'?t|is not) (?:not )?worth (?:living|it)\b",
        r"\b(?:wrote|writing|written) (?:a|my) (?:suicide|goodbye) note\b",
        r"\b(?:this is|these are) my (?:last|final) (?:message|words|goodbye)\b",
        r"\b(?:bought|got|have|saved|stockpiled|collected) (?:a |the |some |enough )?(?:rope|pills|gun|tablets|blades?)\b[^.?!\n]{0,40}\b(?:end|die|kill|tonight|ready)\b",
        r"\bplan(?:ned)? (?:how|when) (?:to|i'?ll|i will) (?:kill my ?self|die|end (?:it|my life))\b",
        r"\bgoing to end it (?:all|tonight|today)\b",
        r"\bsaying goodbye (?:to everyone|forever)\b",
        r"\btired of (?:living|being alive|existing)\b",
        # v2 (from dev-set errors, generalised)
        r"\b(?:what'?s|what is|is there) (?:the|any) point (?:of|in) (?:living|life|going on|being alive|carrying on)\b",
        r"\b(?:sleep|disappear|be gone) (?:forever|for good|permanently)\b",
        r"\bthinking about how (?:i'?d|i would|to) (?:do it|end it)\b",
        r"\b(?:giving|gave) away (?:my|all my) (?:things|stuff|belongings)\b",
        r"\b(?:letters?|notes?) (?:to|for) (?:my )?(?:family|parents|friends)\b[^.?!\n]{0,40}\b(?:why|go|gone|goodbye)\b",
        r"\bdon'?t (?:deserve|want) to (?:live|exist|be alive)\b",
        r"\b(?:won'?t|will not) be (?:around|here|alive|a problem for anyone)\b[^.?!\n]{0,20}\b(?:long|longer|tomorrow|soon)\b",
        r"\b(?:tomorrow|tonight) i (?:won'?t|will not) be (?:alive|here)\b",
        r"\bi'?m done\.?\s*goodbye\b",
        r"\b(?:want|need) (?:the pain|it all|everything) to (?:stop|end) (?:permanently|forever|for good)\b",
        r"\b(?:already )?(?:taken|took|swallowed) (?:a lot of|all (?:my|the)|too many|loads of) (?:pills|tablets|meds)\b",
    ],
    "self_harm": [
        rf"{_I}[^.?!\n]{{0,30}}\b(?:cut|cutting|burn(?:ing|ed|t)?|hurt(?:ing)?|harm(?:ing|ed)?|scratch(?:ing|ed)?|hit(?:ting)?|starv(?:e|ing))\b ?my ?self\b",
        r"\b(?:want|urge|need|going) to (?:cut|hurt|harm|burn) my ?self\b",
        r"\bself[- ]?harm(?:ing|ed)? (?:again|every|today|tonight|last night)\b",
        rf"{_I}[^.?!\n]{{0,20}}\bself[- ]?harm(?:ing|ed)?\b",
        r"\brelapsed (?:on|into) (?:cutting|self[- ]?harm)\b",
        r"\b(?:been|keep|started) (?:hurting|harming|cutting|burning) (?:my ?self|my (?:arms|legs|wrists|skin))\b",
        r"\bhurt my ?self on purpose\b",
    ],
    "harm_others": [
        rf"{_I}[^.?!\n]{{0,30}}\b(?:want|wanna|going|gonna|plan(?:ning)?|will|ready) to (?:kill|hurt|harm|shoot|stab|attack|poison|beat)\b (?:him|her|them|someone|somebody|people|everyone|my\b)",
        r"\b(?:kill|shoot|stab) (?:everyone|them all|my (?:boss|teacher|parents?|mom|dad|wife|husband|girlfriend|boyfriend))\b",
    ],
    "abuse": [
        r"\b(?:he|she|they|my (?:dad|father|mom|mother|husband|wife|partner|boyfriend|girlfriend|uncle|stepdad|stepfather|brother|teacher)) (?:hits|beats|hit|beat|rapes|raped|abuses|abused|chokes|choked|threatens|threatened|touches|touched|molests|molested) me\b",
        r"\bi(?:'m| am|m) (?:being|getting) (?:abused|beaten|hit|raped|molested|hurt) (?:by|at)\b",
        r"\b(?:sexually|physically|emotionally) abus(?:ed|ing) me\b",
        r"\bi(?:'m| am) (?:not safe|scared for my (?:life|safety)) (?:at home|with (?:him|her|them))\b",
        r"\b(?:he|she|they)(?:'s| is| are| will| is going to) (?:going to )?kill me\b",
    ],
    "third_party": [
        r"\bmy (?:friend|brother|sister|son|daughter|partner|mom|mum|mother|dad|father|wife|husband|boyfriend|girlfriend|roommate|classmate|colleague|student|child|kid)\b[^.?!\n]{0,40}\b(?:wants? to|is going to|said (?:she|he|they) (?:wants?|will)|talks? about|threaten(?:s|ed|ing)? to|tried to) (?:kill (?:him|her|them)self|die|end (?:his|her|their) life|commit suicide|hurt (?:him|her|them)self)",
        r"\b(?:someone|somebody|a friend) (?:i know |close to me )?(?:is|has been) (?:suicidal|self[- ]?harming|cutting)\b",
    ],
    "method_request": [
        r"\bhow (?:many|much)\b[^?\n]{0,60}\b(?:to|would|will|does it take to|can) (?:die|kill|be fatal|be lethal|overdose|od|end (?:it|my life))\b",
        r"\b(?:lethal|fatal|deadly|toxic) (?:dose|amount|quantity)\b",
        r"\b(?:best|easiest|quickest|fastest|painless|least painful|surest|most effective|quick) (?:way|method|methods|ways) (?:to|of) (?:die|kill (?:my|your|one)?self|commit suicide|suicide|end (?:it|my life|your life))\b",
        r"\bhow (?:to|do i|do you|can i|could i|would i) (?:kill (?:my|your)?self|commit suicide|hang (?:my|your)?self|overdose|slit (?:my|your) wrists?|end my life)\b",
        r"\bhow (?:to|do i|can i) (?:cut|hurt|harm|burn) (?:my|your)self\b",
        r"\b(?:which|what) (?:pills|drugs|medications?|medicines?|tablets|poison|chemicals?) (?:can|could|will|would|should i take to) (?:kill|end|die|overdose)\b",
        r"\bhow (?:long|much time) does it take to (?:die|bleed out|overdose)\b",
        r"\bhide (?:my |the )?(?:cuts|self[- ]?harm|scars) from\b[^?\n]{0,30}\b(?:so|without)\b",
        # v2 (generalised from dev-set errors)
        r"\b(?:suicide|self[- ]?harm) methods?\b|\bmethods? (?:of|for) (?:suicide|self[- ]?harm|killing (?:my|your)self)\b",
        r"\bhide (?:my |the |an )?(?:self[- ]?harm|cuts|scars|eating disorder|anorexia|bulimia)\b",
        r"\b(?:starve|purge) my ?self\b[^?\n]{0,40}(?:fast|quick|without|so)\b",
    ],
}

# Structural method-request detector (v2): a question that combines a lethality word with a method/quantity word,
# e.g. "How many mg of amitriptyline is lethal?", "what household chemicals can kill a person".
_QUESTION = re.compile(r"^\s*(?:how|what|which|where|is|are|can|could|would|will|does|do|tell me|give me|list)\b|\?\s*$", re.I)
_LETHAL = re.compile(r"\b(?:lethal|fatal|deadly|kill(?:s|ed)?|die|dying|death|overdose|od)\b", re.I)
_METHOD = re.compile(
    r"\b(?:dose|doses|amount|mg|milligrams?|pills?|tablets?|medications?|meds|drugs?|chemicals?|poison|insulin|"
    r"paracetamol|acetaminophen|tylenol|ibuprofen|how many|how much|how long|how high|how deep|high|deep|"
    r"cut|slit|hang|hanging|jump|jumping|bridge|rope|method|methods|way|ways|instructions|steps?)\b",
    re.I,
)
_INFO_SAFE = re.compile(r"\b(?:warning signs?|signs of|help|support|prevent|prevention|cope|coping|safety plan|risk factors?)\b",
                        re.I)


def structural_method_request(text: str) -> bool:
    t = _clean(text)
    return bool(_QUESTION.search(t) and _LETHAL.search(t) and _METHOD.search(t) and not _INFO_SAFE.search(t))


NEGATION_GUARD = re.compile(
    r"\b(?:i'?m not|i am not|im not|i'?ve never|i have never|never (?:been|felt|wanted)|not (?:feeling )?suicidal|"
    r"don'?t want to (?:kill|hurt|harm) my ?self|no longer (?:feel|feeling)|not going to (?:kill|hurt))\b",
    re.I,
)

_COMPILED = {k: [re.compile(p, re.I) for p in v] for k, v in PATTERNS.items()}
CRISIS_CATEGORIES = ("suicidal", "self_harm", "harm_others", "abuse")


@dataclass
class LexiconResult:
    categories: dict[str, list[str]] = field(default_factory=dict)  # category -> matched snippets
    negated: bool = False
    has_negation: bool = False  # an explicit risk negation anywhere in the message ("I'm not suicidal")

    @property
    def crisis(self) -> bool:
        return any(c in self.categories for c in CRISIS_CATEGORIES)

    @property
    def method_request(self) -> bool:
        return "method_request" in self.categories

    @property
    def third_party(self) -> bool:
        return "third_party" in self.categories


def _clean(text: str) -> str:
    return text.replace("’", "'").replace("‘", "'").lower()


def scan(text: str) -> LexiconResult:
    t = _clean(text)
    res = LexiconResult()
    res.has_negation = bool(NEGATION_GUARD.search(t)) or bool(
        re.search(r"\b(?:not|never|no longer)\b[^.?!\n]{0,20}\b(?:suicidal|going to hurt|hurt(?:ing)? my ?self|self[- ]?harm)", t)
    ) or bool(re.search(r"\b(?:years? ago|as a (?:teen|teenager|kid|child)|in the past|used to)\b", t))
    for cat, pats in _COMPILED.items():
        for p in pats:
            m = p.search(t)
            if m:
                res.categories.setdefault(cat, []).append(m.group(0)[:120])
    if "method_request" not in res.categories and "third_party" not in res.categories and structural_method_request(text):
        res.categories["method_request"] = ["(structural)"]
    # Negation guard: drop suicidal/self-harm hits that sit in a sentence with an explicit negation, but only
    # when no other (un-negated) sentence also matches.
    if any(c in res.categories for c in ("suicidal", "self_harm")):
        sentences = re.split(r"(?<=[.!?\n])\s+", t)
        hit_sentences = [s for s in sentences if any(p.search(s) for c in ("suicidal", "self_harm") for p in _COMPILED[c])]
        if hit_sentences and all(NEGATION_GUARD.search(s) for s in hit_sentences):
            res.negated = True
            res.categories.pop("suicidal", None)
            res.categories.pop("self_harm", None)
    return res
