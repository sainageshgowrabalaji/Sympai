"""
Step 4. The triage desk.

Every message the person sends passes through here before any model sees it.
It has one job. Catch emergencies, and catch attempts to trick the agents.

Why in code and not in the prompt? In testing, the coordinator first sent
ear pain to 911, and after a prompt fix it sent stroke signs to urgent care.
Tuning a prompt around a safety rule just moves the mistake around. Code
says the same fixed words every time, runs in microseconds, costs no tokens
and cannot be talked out of it.

Layer 1  Fixed rules (this file). Precise and instant, but they only know the
         phrasings we wrote down.
Layer 2  A SetFit classifier (classifier.py). It learns from examples, so it
         catches paraphrases the rules miss.
Layer 3  Llama Prompt Guard 2 on Groq. It scores how likely a message is an
         attempt to override the agent's instructions.
"""

import re
from dataclasses import dataclass, field

from sympai.classifier import THRESHOLD as CLASSIFIER_THRESHOLD

# ---------------------------------------------------------------- layer 1

CALL_911 = "This could be an emergency. Please call 911 now, or ask someone near you to call."
AFTER_911 = (
    "Do not drive yourself.\n\n"
    "If this was about something in the past or about someone else, tell me and we will carry on."
)


@dataclass(frozen=True)
class Rule:
    id: str
    level: str  # "911", "poison" or "988"
    patterns: tuple[str, ...]
    reason: str


RULES = (
    Rule(
        "stroke",
        "911",
        (
            r"face (is |was |keeps )?droop",
            r"slurr",
            r"can.?t (speak|talk|get (my |the )?words out)",
            r"trouble (speaking|talking)",
            r"sudden(ly)? .{0,40}(weak|numb|paraly|confus|blind|vision|double vision)",
            r"(one|left|right) side of (my|his|her|the) (body|face)",
            r"can.?t (move|feel) (my |his |her )?(arm|leg|face|hand)",
            r"(lost|losing|loss of) (my |his |her )?vision",
        ),
        "Sudden weakness or numbness, a drooping face, trouble speaking or sudden vision loss "
        "can be signs of a stroke. Stroke treatment works best in the first hours, and urgent care cannot give it.",
    ),
    Rule(
        "worst_headache",
        "911",
        (r"worst headache", r"thunderclap", r"headache.{0,40}thunder", r"sudden(ly)? .{0,20}(severe|terrible|explosive) headache"),
        "A sudden, severe headache, the worst of your life, can mean bleeding in the brain and needs the ER now.",
    ),
    Rule(
        "chest",
        "911",
        (r"chest (pain|pressure|tightness|is tight|feels tight|hurts|squeez)", r"heart attack", r"pain .{0,30}(left arm|jaw)"),
        "Chest pain or pressure can be a heart attack. Call even if you are not sure, and do not wait to see if it passes.",
    ),
    Rule(
        "breathing",
        "911",
        (r"can.?t breathe", r"cannot breathe", r"(trouble|difficulty|struggling) breathing", r"hard to breathe",
         r"short(ness)? of breath", r"gasping", r"choking"),
        "Trouble breathing needs help right away.",
    ),
    Rule(
        "allergic",
        "911",
        (r"throat .{0,15}(closing|swell|swollen)", r"(lips?|tongue|face|mouth) .{0,15}(swell|swollen)", r"anaphyla"),
        "Swelling of the throat, lips or tongue can be a severe allergic reaction. "
        "If there is an epinephrine auto-injector, use it, then call 911.",
    ),
    Rule(
        "collapse",
        "911",
        (r"passed out", r"fainted", r"unconscious", r"unresponsive", r"won.?t wake up", r"seizure", r"convuls"),
        "Fainting, a seizure or someone not waking up needs emergency help.",
    ),
    Rule(
        "bleeding",
        "911",
        (r"bleeding (heavily|a lot|badly|won.?t stop)", r"can.?t stop (the )?bleeding",
         r"(coughing|vomiting|throwing) up blood"),
        "Heavy bleeding, or coughing or vomiting blood, needs emergency help. "
        "Press firmly on a bleeding wound while you wait.",
    ),
    Rule(
        "poison",
        "poison",
        (r"overdos", r"took too many (pills|tablets|meds)", r"swallowed .{0,20}(bleach|poison|chemical|detergent|battery)",
         r"(?<!food )poison(ed|ing)"),
        "Call Poison Control at 1-800-222-1222. It is free and open 24 hours. "
        "If the person is unconscious, having a seizure or struggling to breathe, call 911 first.",
    ),
    Rule(
        "crisis",
        "988",
        (r"kill myself", r"end (my (own )?life|it all)", r"suicid", r"want to die", r"don.?t want to (live|be alive|be here)",
         r"(want|wanted|going|thinking about|thoughts of|urge) (to )?(hurt|harm|cut)(ing)? myself", r"self[- ]?harm",
         r"(started|been|keep|kept|begun) (hurting|harming|cutting) myself", r"decided to end",
         r"end it (all|tonight|today|this week|this weekend|soon)",
         r"no reason to live", r"better off without me"),
        "I'm really glad you told me, and you deserve support right now. "
        "You can call or text 988 to reach the Suicide and Crisis Lifeline, any time, day or night. "
        "If you are in immediate danger or might act on these thoughts, call 911. I'm here to keep talking too.",
    ),
)

# Words that cancel a match when they come before it in the same clause, as in
# "no weakness or numbness".
NEGATIONS = {
    "no", "not", "never", "without", "denies", "deny", "none", "nothing", "neither", "nor",
    "dont", "don't", "doesnt", "doesn't", "didnt", "didn't", "havent", "haven't",
    "hasnt", "hasn't", "isnt", "isn't", "arent", "aren't", "wasnt", "wasn't",
}
# A negation covers the rest of its clause, so it reaches every item in a
# list like "I don't feel shortness of breath, dizziness, or pain in my arm".
# An earlier version only looked 5 words back, and in the scenario run that
# exact answer to SympAI's own question set off a false 911 alarm.
#
# So where a clause ends matters. It ends at a full stop, at "but" and its
# cousins, and where a new subject starts, as in "I haven't slept and my chest
# hurts" or "I don't have a fever, my chest hurts". A comma before a plain noun,
# as in "no weakness, numbness, or trouble speaking", keeps the list together.
_SUBJECT = r"(?:i|my|it|he|she|there|now)\b"
_CLAUSE_BREAKS = re.compile(
    r"[.;!?\n]|\bbut\b|\bhowever\b|\balthough\b|\bthough\b|\band now\b"
    rf"|,?\s*\band\s+(?={_SUBJECT})|,\s*(?={_SUBJECT})"
)
# Only when punctuation follows. "No weakness" is a negation, "No, I also..." is an answer.
_ANSWER_WORD = re.compile(r"^\s*(no|nope|yes|yeah|yep)\s*[,.!-]+\s*")
_COMPILED = [(rule, [re.compile(p) for p in rule.patterns]) for rule in RULES]


def _negated(clause: str, start: int) -> bool:
    words_before = re.findall(r"[a-z']+", clause[:start])
    return any(word in NEGATIONS for word in words_before)


def matching_rules(text: str) -> list[Rule]:
    """Every rule that fires on this message, most urgent first."""
    fired = []
    # Mac keyboards type a curly apostrophe in don’t, which would hide the negation.
    normalized = text.lower().replace("\u2019", "'")
    for clause in _CLAUSE_BREAKS.split(normalized):
        # "No, I also have tinnitus" is an answer word, not a negation of what follows.
        clause = _ANSWER_WORD.sub("", clause)
        for rule, patterns in _COMPILED:
            if rule in fired:
                continue
            for pattern in patterns:
                match = pattern.search(clause)
                if match and not _negated(clause, match.start()):
                    fired.append(rule)
                    break
    return sorted(fired, key=lambda r: RULES.index(r))


def emergency_reply(rules: list[Rule]) -> str:
    """The fixed words shown for an emergency. No model writes these."""
    levels = {r.level for r in rules}
    reasons = "\n\n".join(r.reason for r in rules)
    if "911" in levels:
        return f"{CALL_911}\n\n{reasons}\n\n{AFTER_911}"
    return reasons


# ---------------------------------------------------------------- layer 3

PROMPT_GUARD_MODEL = "meta-llama/llama-prompt-guard-2-86m"
INJECTION_THRESHOLD = 0.9
INJECTION_REPLY = (
    "That message looks like an attempt to change how I work, so I will not act on it. "
    "If you meant it as a health question, try asking it another way."
)


def parse_guard_output(content: str | None) -> float | None:
    """Prompt Guard's reply as a 0 to 1 attack score.

    Groq's docs do not show the exact reply format, so this accepts a number
    or a label. scripts/check_prompt_guard.py prints the raw reply so we can
    see for ourselves.
    """
    text = (content or "").strip()
    try:
        return min(max(float(text), 0.0), 1.0)
    except ValueError:
        pass
    upper = text.upper()
    if any(word in upper for word in ("MALICIOUS", "JAILBREAK", "INJECTION")):
        return 1.0
    if "BENIGN" in upper:
        return 0.0
    return None


def prompt_guard_score(text: str) -> float | None:
    """Ask Prompt Guard 2 how likely this message is an attack. None if it could not answer.

    It fails open, meaning a network error lets the message through. That is
    safe here because the emergency rules above never touch the network.
    The model reads at most 512 tokens, so long messages are cut.
    """
    try:
        from groq import Groq  # the plain Groq SDK. Not everything needs LangChain.

        reply = Groq(max_retries=2).chat.completions.create(
            model=PROMPT_GUARD_MODEL,
            messages=[{"role": "user", "content": text[:1500]}],
        )
        return parse_guard_output(reply.choices[0].message.content)
    except Exception:
        return None


# ---------------------------------------------------------------- the desk


@dataclass
class TriageResult:
    level: str  # "clear", "urgent", "911", "poison", "988" or "injection"
    source: str | None = None  # "rules", "classifier" or "guard"
    rules: list[str] = field(default_factory=list)
    injection_score: float | None = None
    reply: str | None = None  # the words the desk shows, if any
    past: bool = False  # the warning signs seem to be about something that already happened

    @property
    def alert_keys(self) -> list[str]:
        """What counts as 'already warned about this' in a conversation.

        A past episode gets its own key, so mentioning an old chest pain does
        not use up the alarm for a chest pain happening right now.
        """
        if self.source == "rules":
            return [f"{rule}-past" if self.past else rule for rule in self.rules]
        return [f"{self.source}-{self.level}" + ("-past" if self.past else "")]


# People describe what already happened to them, as you did in testing. Those
# words should not trip the full emergency stop, but we still check it is not
# happening again right now.
_PAST = re.compile(
    r"\b(last (week|month|year|summer|winter)|(\d+|a|one|two|three|few|couple of|many|several) "
    r"(days?|weeks?|months?|years?) (ago|back)|years? back|in (19|20)\d\d|used to|previously|in the past|"
    r"history of|once had|suffered|when i was|i had .{0,40} before)\b"
)
def _is_past(text: str) -> bool:
    return bool(_PAST.search(text.lower().replace("\u2019", "'")))


PAST_NOTE = (
    "It sounds like this happened before. If any of it is happening right now, call 911.\n\n"
    "Otherwise, let me ask a couple of questions about it."
)


# In testing the classifier fired on three chest messages in a row and showed
# the same 911 notice every time. The person could never get past it to say
# more. A safety check that cries wolf gets ignored, which hospitals call
# alarm fatigue. So a classifier hit now WARNS and lets the coordinator ask
# whether it is happening right now. Only an exact rule match stops the turn.
WARN_911 = (
    "What you describe can sometimes be an emergency. If it is happening right now and it is severe, "
    "came on suddenly, or comes with trouble breathing, sweating, nausea or fainting, call 911 now.\n\n"
    "Otherwise, let me ask a couple of quick questions."
)
CLASSIFIER_NOTES = {
    "911": WARN_911,
    "988": next(rule.reason for rule in RULES if rule.id == "crisis"),
    "urgent": None,  # no notice. The coordinator is told to aim for same-day care instead.
}


def check(text: str, guard=None, classifier=None, threshold: float | None = None) -> TriageResult:
    """Run the three layers in order. The graph passes the real guard and classifier, tests pass fakes."""
    fired = matching_rules(text)
    if fired:
        # Emergencies win over everything, and skip the network call entirely.
        ids = [r.id for r in fired]
        if _is_past(text) and fired[0].level != "988":
            return TriageResult(level=fired[0].level, source="rules", rules=ids, reply=PAST_NOTE, past=True)
        return TriageResult(level=fired[0].level, source="rules", rules=ids, reply=emergency_reply(fired))
    found = classifier(text) if classifier else None
    if found:
        label, confidence = found
        if label != "clear" and confidence >= (CLASSIFIER_THRESHOLD if threshold is None else threshold):
            past = label != "988" and _is_past(text)
            reply = PAST_NOTE if past and label == "911" else CLASSIFIER_NOTES[label]
            return TriageResult(
                level=label, source="classifier", rules=[f"classifier {confidence:.2f}"], reply=reply, past=past
            )
    score = guard(text) if guard else None
    if score is not None and score >= INJECTION_THRESHOLD:
        return TriageResult(level="injection", source="guard", injection_score=score, reply=INJECTION_REPLY)
    return TriageResult(level="clear", injection_score=score)
