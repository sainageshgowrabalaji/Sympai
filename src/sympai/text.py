"""Small text helpers shared by every part of SympAI."""

import re


def as_text(content) -> str:
    """A message's words, whatever shape the content came in.

    The terminal chat sends a plain string. Chat screens, like Agent Chat UI,
    send a list of content blocks such as [{"type": "text", "text": "..."}],
    so they can carry images and PDFs too. Everything that reads a message
    goes through this, so both shapes work.
    """
    if isinstance(content, str):
        return content
    parts = []
    for part in content or []:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type") == "text":
            parts.append(part.get("text", ""))
    return "\n".join(p for p in parts if p)


def has_attachment(content) -> bool:
    """True when a message carries something other than text, like an image or a PDF."""
    return isinstance(content, list) and any(
        isinstance(part, dict) and part.get("type") not in (None, "text") for part in content
    )

# gpt-oss was trained with its own citation marks like 【2†L2-L9】. They point
# at nothing in our app. The prompt asks for plain URLs, and this removes any
# that slip through anyway. Anything that must ALWAYS hold is enforced in
# code, not only in a prompt.
_TRAINING_CITATIONS = re.compile(r"\s*【[^】]*】")

# Models often write a non-breaking hyphen (U+2011) or a dash where a plain
# hyphen belongs, as in "Same‑day". A check looking for "same-day" then misses
# it. Every check below reads text through this first.
_HYPHENS = re.compile("[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]")


def _plain(text: str) -> str:
    return _HYPHENS.sub("-", text)


# The backup model once wrote "you can find one with `find_clinicians` if you
# give a ZIP code". Tool names mean nothing to a person, so code swaps them for
# plain words, whatever the prompt said.
_TOOL_WORDS = {
    "find_clinicians": "a clinician search",
    "care_options": "a care search",
    "search_medlineplus": "MedlinePlus",
    "search_pubmed": "a PubMed search",
    "search_clinical_trials": "a ClinicalTrials.gov search",
    "consult_department": "a specialist consult",
}
_TOOL_NAMES = re.compile(r"`?\b(" + "|".join(_TOOL_WORDS) + r")\b(\(\))?`?")


def clean_answer(text: str) -> str:
    text = _TRAINING_CITATIONS.sub("", text or "")
    return _TOOL_NAMES.sub(lambda m: _TOOL_WORDS[m.group(1)], text).strip()


# In three test runs the coordinator kept asking two questions at once, like
# "How old are you, and do you have any medical conditions?", even with a rule
# and an example in the prompt. So code now keeps only the first question.
# This is an output guardrail, the mirror image of triage, which guards input.
_SECOND_QUESTION = re.compile(
    r",?\s+(and|also)\s+(have|has|had|do|does|did|is|are|was|were|can|could|how|what|when|where|which|any)\b"
    # The backup model asked "Do you exercise or do you often feel dehydrated or have
    # low potassium?", three questions joined with "or". "Sharp, cramping, or achy?" is
    # still one question, so only "or" followed by a new "do you" style question counts.
    r"|,?\s+or\s+(do|does|did|have|has|are|is|was|were|can|could)\s+you\b",
    re.IGNORECASE,
)


def one_question(text: str) -> str:
    """If a short reply asks two questions, keep the first. Longer answers are left alone.

    A kind opening line before the question is fine. In the chat screen test,
    "I'm sorry... frustrating.\n\nWhen did it start, and how long did it last?"
    slipped through, because the line break made the whole reply look like a
    long answer. So the last paragraph is checked on its own.
    """
    paragraphs = text.strip().split("\n\n")
    opening = paragraphs[0]
    is_kind_line = "?" not in opening and "**" not in opening and "\n" not in opening and len(opening.split()) <= 30
    if len(paragraphs) == 2 and is_kind_line:
        return paragraphs[0] + "\n\n" + one_question(paragraphs[1])
    reply = text.strip()
    if not reply.endswith("?") or "\n" in reply or "**" in reply or len(reply.split()) > 60:
        return text
    if reply.count("?") > 1:
        first = reply[: reply.index("?") + 1]
        return first if len(first.split()) >= 3 else text
    match = _SECOND_QUESTION.search(reply)
    if match and len(reply[: match.start()].split()) >= 3:
        return reply[: match.start()].rstrip(" ,") + "?"
    return text


# In the scenario run SympAI wrote "Urgent care, go if the heaviness becomes
# crushing, or comes with shortness of breath or sweating". Those are heart
# attack signs, and urgent care cannot treat a heart attack. The prompt already
# says so, and the model still slipped. So code checks every line that mentions
# urgent care, and swaps one that lists a serious warning sign for a 911 line.
_SERIOUS_SIGNS = {
    "crushing": "crushing chest pain or pressure",
    "chest pressure": "chest pressure",
    "shortness of breath": "shortness of breath",
    "short of breath": "shortness of breath",
    "trouble breathing": "trouble breathing",
    "sweating": "sweating",
    "faint": "fainting",
    "passing out": "fainting",
    "slurred": "slurred speech",
    "droop": "a drooping face",
    "weakness on one side": "weakness on one side",
    "worst headache": "the worst headache of your life",
}


def misrouted_signs(line: str) -> list[str]:
    """Serious warning signs that this line sends to urgent care instead of 911."""
    # "urgent-care" with a hyphen slipped past the first version of this check.
    low = re.sub(r"urgent[\s-]+care", "urgent care", _plain(line).lower())
    # A line that says urgent care cannot help, like the triage desk's own stroke notice, is fine.
    if "urgent care" not in low or "911" in low or any(w in low for w in ("cannot", "can't", "not ", "never")):
        return []
    found = []
    for words, sign in _SERIOUS_SIGNS.items():
        if words in low and sign not in found:
            found.append(sign)
    return found


def route_serious_signs_to_911(text: str) -> tuple[str, bool]:
    """Replace any line that sends a serious warning sign to urgent care. Returns the text and whether it changed."""
    lines, changed = text.split("\n"), False
    for i, line in enumerate(lines):
        signs = misrouted_signs(line)
        if signs:
            bullet = re.match(r"^\s*(?:[-*]\s+|\d+\.\s+)?", line).group(0)
            listed = ", ".join(signs[:-1]) + (" or " if len(signs) > 1 else "") + signs[-1]
            lines[i] = f"{bullet}**Call 911, not urgent care**, if you have {listed}. Urgent care cannot treat a heart attack or a stroke."
            changed = True
    return "\n".join(lines), changed



# When the triage desk has flagged a conversation as urgent or a possible
# emergency, the final answer must not route below that. In the scenario run
# the answer led with self-care and never said when to call 911. So code adds
# the two lines when they are missing. Fixed words, the same every time.
SAME_DAY_LINE = (
    "**Get checked soon.** Because of what you described, see a doctor today or tomorrow, "
    "at primary care, telehealth or urgent care."
)
CALL_911_LINE = (
    "**Call 911** if it becomes severe or sudden, or comes with chest pain or pressure, "
    "trouble breathing, sweating, fainting, or weakness on one side."
)
_SAME_DAY = re.compile(r"same[\s-]day|today|tomorrow|within (a|one|1|24|48) (day|hours)|urgent[\s-]care", re.I)


def enforce_minimum_care(text: str, peak: str | None) -> tuple[str, bool]:
    """Add the same-day and 911 lines to an analysis when the triage desk flagged the conversation."""
    if peak not in ("urgent", "911") or "Most likely" not in text:
        return text, False
    extra = []
    if not _SAME_DAY.search(_plain(text)):
        extra.append(SAME_DAY_LINE)
    if "911" not in text:
        extra.append(CALL_911_LINE)
    return (text.rstrip() + "\n\n" + "\n\n".join(extra), True) if extra else (text, False)
