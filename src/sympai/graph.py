"""
Steps 3 and 4. The hospital as a graph.

Nodes (the people in the hospital)
    triage        checks every new message for emergencies and tricks, in code
    coordinator   talks to the person and decides what happens next
    department    runs consult_department calls and files typed reports
    research      runs MedlinePlus, PubMed and ClinicalTrials.gov lookups
    services      finds real clinicians and care options near a ZIP code
    remember      updates the person's chart after each turn

Edges (who hands off to whom)
    START       -> triage
    triage      -> END with fixed emergency words, or on to the coordinator
    coordinator -> department and/or research, when it asks for them
    coordinator -> END, when it writes to the person (a question or an answer)
    department  -> coordinator
    research    -> coordinator

This is the loop you wrote by hand in step 2, drawn as a graph. The
difference is shared state. Every node reads the whole case and returns only
what it changed, and a reducer decides how that change merges in.
"""

import logging
import operator
import os
import re
import time
from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from langgraph.graph import END, START, MessagesState, StateGraph

from sympai.departments import MAX_RETRIES, consult, departments_for_coordinator, load_departments
from sympai.text import (
    as_text,
    clean_answer,
    enforce_minimum_care,
    has_attachment,
    one_question,
    route_serious_signs_to_911,
)
from sympai.tools import (
    care_options,
    find_clinicians,
    search_clinical_trials,
    search_medlineplus,
    search_pubmed,
)
from sympai import chart as charts
from sympai.classifier import classify
from sympai.triage import check, prompt_guard_score

COORDINATOR_MODEL = "openai/gpt-oss-120b"
# Groq's free tier gives the 120b model 200,000 tokens a day, and one long
# conversation can use 30,000 of them. When they run out, the 20b model takes
# over. It has its own daily allowance, and every safety check in code still runs.
BACKUP_COORDINATOR_MODEL = "openai/gpt-oss-20b"
BACKUP_NOTE = "_The main model has used up its free allowance for now, so SympAI's backup model answered this one._"
DEPARTMENT_NAMES = {"ent": "ENT", "heart_lungs": "heart and lungs", "bones_muscles": "bones and muscles"}
STATUS_LINES = {
    "search_medlineplus": "looking this up on MedlinePlus",
    "search_pubmed": "searching PubMed for recent reviews",
    "search_clinical_trials": "searching ClinicalTrials.gov",
    "find_clinicians": "looking up clinicians near {zip_code}",
    "care_options": "looking up care options near {zip_code}",
}


def status_line(tool_calls: list[dict]) -> str:
    """What SympAI is doing, in words, for a message that only calls tools.

    The chat screen showed those messages as empty bubbles. The model never
    sees this text, context_for_model sends it an empty message as before.
    """
    parts = []
    for call in tool_calls:
        args = call.get("args", {}) or {}
        if call["name"] == "consult_department":
            key = str(args.get("department", "")).strip().lower()
            parts.append(f"asking the {DEPARTMENT_NAMES.get(key, key.replace('_', ' '))} department")
        elif call["name"] in STATUS_LINES:
            parts.append(STATUS_LINES[call["name"]].format(zip_code=args.get("zip_code", "you")))
    if not parts:
        return ""
    joined = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    return f"_{joined[0].upper()}{joined[1:]}..._"


SORRY_REPLY = """Something went wrong on my side while writing that answer, so I could not finish it. Please send your message again.

If this cannot wait, call your doctor's office or go to urgent care today. For chest pain, trouble breathing, signs of a stroke or anything that feels like an emergency, call 911. If you are thinking about harming yourself, call or text 988."""
NO_TOOLS_NOTE = "Your lookups are not working for this reply. Do not try to use them. Answer with what you already know, or ask your next question, and say plainly if there is something you could not check."
log = logging.getLogger("sympai")

BUSY_REPLY = """SympAI has used up its free allowance for now, so it cannot give you a careful answer. Please try again in about {wait}.

If this cannot wait, call your doctor's office or go to urgent care today. For chest pain, trouble breathing, signs of a stroke or anything that feels like an emergency, call 911. If you are thinking about harming yourself, call or text 988."""

# Version 2 of the prompt. Every rule below exists because a real test run
# broke it. Rules added in 3.1 are marked with the problem they fix.
COORDINATOR_PROMPT = """You are SympAI, the coordinator of a small hospital of AI specialists.
You help one person understand a health problem and find the right care in the US.
You are not a doctor, and you never say a diagnosis is confirmed.

Emergencies
Keep 911 for real emergencies, like trouble breathing, chest pain, fainting, or stroke signs such as sudden weakness, numbness, slurred speech or double vision. Say it first, before anything else.
Chest pain or pressure that is crushing, lasts more than a few minutes, or comes with shortness of breath, sweating, nausea or fainting is always 911, never urgent care.
For stroke signs or a sudden, worst-ever headache, always say call 911. Never send them to urgent care, which cannot treat a stroke.
Problems that need a doctor soon but are not life threatening go to urgent care or a same-day appointment, not 911.

Questions
If the person names a symptom, assume they have it and start asking.
They may be asking for someone else, such as a parent, a child or a friend. Then ask about that person, and give the advice for them.
Ask exactly one short question per turn, as a plain sentence with no heading, then stop and wait. Never put two questions in one.
For example, ask "When did the pain start?" and not "When did it start, and how bad is it?"
Ask only what would change your advice, such as when it started, how bad it is, what else they notice, their age group, and the warning signs for this symptom.
Use everyday words. Never ask about lab results or terms a person would not know, like potassium or magnesium levels. Ask what they would notice instead, like how much water they drink or whether it comes after exercise.
Three to five questions is usually enough. If they ask where to go, still ask what you need to pick the right level of care.
For thoughts of suicide or self-harm, the number to give is 988, by call or text. 911 is only for immediate danger.
For stress, low mood, sleep problems or anxiety, be warm first. Say something kind about how hard it sounds before your first question, and ask gently how they are coping.

Analysis
When more answers would not change your advice, consult the department that fits with consult_department, and look facts up with search_medlineplus.
If the case spans two fields, such as calf pain with chest pain, consult both departments in the same turn.
For a rare or long-lasting condition, or when they ask what research says, use search_pubmed for recent reviews and search_clinical_trials for studies recruiting now.
Label research honestly. A review sums up studies, one paper is not settled science, and a trial is a study, not a treatment.
If a department you need does not exist yet, say so plainly and give your own careful read.
If they describe something from the past that was never checked, suggest bringing it up at a routine primary care visit.
Then answer with these parts, each a short bold heading followed by one or two sentences.
   Most likely
   Also possible
   How sure I am (low, medium or high, and why)
   What would confirm it
   Where to go (self-care and OTC, pharmacist, primary care or telehealth, urgent care, a specialist, or the ER)
   Sources (the plain MedlinePlus URLs you used)

Medicines
Suggest an OTC medicine only after you know about allergies, other medicines they take, pregnancy, and kidney, liver or stomach problems. If you do not know, ask first.
For doses, say to follow the label. Prescription (Rx) medicines are only ever something to ask a doctor about.

Places and costs
To suggest real doctors, use find_clinicians with the kind of doctor and their ZIP code. For urgent care, pharmacies, low-cost clinics or telehealth, use care_options. Ask for the ZIP code if you do not have it.
Name a clinic, doctor, address or phone number only if one of those tools returned it, and copy it exactly. Never name a distance, transit route, app or price.
Always add that they should check with their insurer or the office that the doctor takes their plan.
If they ask how to see a specialist, like a neurologist, explain the usual path. Many plans let people book a specialist directly, while HMO plans need a referral from a primary care doctor first. The plan's member services or website says which applies. Then offer to find that kind of doctor near their ZIP code.
Never show the names of your tools, like find_clinicians. Ask for what you need in plain words, like their ZIP code.
Never state facts about the person's own insurance plan, such as its deductible, coinsurance or referral rules, even when the plan's name seems to suggest them. Explain what the terms mean, and say the real numbers are in the plan's Summary of Benefits and Coverage and with member services, on the number on their card.

Sources
If something has no source from your tools, say it comes from general medical knowledge you could not check.
Never say "studies show", "experts agree" or "consensus" unless a tool gave you that source.
List only public sources, like MedlinePlus pages. A department consult is not a source.

Emergency notices
The triage desk sometimes answers first with a fixed emergency notice, shown as your message.
If the person says it was about the past or about someone else, carry on normally.
If the conversation has touched on suicide or self-harm, stay warm and supportive, keep 988 in view, and let them lead.

Always
Keep each answer under 300 words unless they ask for more detail.
If they ask what a word means, such as "what is OTC", answer it directly.
After your analysis, keep helping with follow-up questions.
Be honest even when the news is worrying, and never make it sound worse than the evidence.
Use plain words, and explain terms like OTC or Rx the first time you use them."""

# The lookup budget. In the visual snow test the coordinator searched and
# consulted six times in one turn and hit the recursion limit without ever
# answering. Now, after this many rounds of tools in one turn, code takes
# the tools away and it has to answer with what it has.
MAX_TOOL_ROUNDS = 3
BUDGET_NOTE = """You have used all your lookups for this turn, so you cannot call any more tools.
Answer now with what you have, and say plainly what you could not check."""


# A custom reducer. Most fields either append (messages) or overwrite (triage).
# This one keeps the MOST SERIOUS level seen so far in the conversation, so a
# warning on turn 1 still counts when the answer is written on turn 6. In the
# scenario run it did not, and the answer forgot the conversation was urgent.
SEVERITY = {"clear": 0, "urgent": 1, "911": 2, "poison": 2, "988": 2}


def keep_most_serious(old: str | None, new: str | None) -> str:
    old, new = old or "clear", new or "clear"
    return new if SEVERITY.get(new, 0) > SEVERITY.get(old, 0) else old


class Consult(MessagesState):
    """The case file every node shares.

    `messages` comes from MessagesState, and its reducer APPENDS new messages
    instead of replacing the list. `reports` uses operator.add, so each
    department report is appended too. Without a reducer, a node's return
    value would overwrite the field.
    """

    reports: Annotated[list[dict], operator.add]
    # No reducer, so each new triage result simply replaces the last one.
    triage: dict
    # Every alert already shown in this conversation, so the same one is not repeated.
    alerts_shown: Annotated[list[str], operator.add]
    # The most serious triage level so far, kept by the reducer above.
    peak: Annotated[str, keep_most_serious]
    # The language Whisper heard, when the person speaks instead of typing.
    language: str


CLEAR = {"level": "clear", "source": None, "rules": [], "injection_score": None, "shown": False,
         "blocking": False, "past": False}


def make_triage(guard, classifier):
    """The triage desk as a node. `guard` and `classifier` are the two checks, or None to skip them.

    What the person sees depends on where the alarm came from.
    - An exact rule match, the first time, shows the fixed notice and ends the turn.
    - A classifier hit shows a short warning, then the coordinator carries on and asks.
    - Anything already warned about in this conversation is not shown again. The
      coordinator still hears about it, through the triage field in the state.
    - A prompt attack is always refused.
    """

    def triage(state: Consult) -> dict:
        last = state["messages"][-1]
        if last.type != "human":
            return {"triage": CLEAR}
        result = check(as_text(last.content), guard=guard, classifier=classifier)
        info = {**CLEAR, "level": result.level, "source": result.source, "rules": result.rules,
                "injection_score": result.injection_score, "past": result.past}
        update = {"triage": info}
        if result.level == "clear":
            return update
        if not result.past and result.level in SEVERITY:
            update["peak"] = result.level  # an old episode does not raise the level for now
        if result.level == "injection":
            info.update(shown=True, blocking=True)
            update["messages"] = [AIMessage(result.reply, response_metadata={"sympai_triage": "injection"})]
            return update

        new_alerts = [key for key in result.alert_keys if key not in state.get("alerts_shown", [])]
        if result.reply and new_alerts:
            info.update(shown=True, blocking=result.source == "rules" and not result.past)
            update["messages"] = [AIMessage(result.reply, response_metadata={"sympai_triage": result.level})]
            update["alerts_shown"] = new_alerts
        return update

    return triage


def after_triage(state: Consult):
    return END if state["triage"].get("blocking") else "coordinator"


# How the triage desk's view reaches the coordinator. Nodes do not call each
# other. One writes to the shared state and the next one reads it.
TRIAGE_HINTS = {
    "911": "The triage desk flagged this message as a possible emergency. First find out whether it is happening "
    "right now and whether warning signs are present. If they are, tell them plainly to call 911.",
    "poison": "The triage desk flagged a possible poisoning. Keep Poison Control, 1-800-222-1222, in view.",
    "urgent": "The triage desk thinks this may need a doctor today. Unless your questions show otherwise, "
    "point to same-day care, such as urgent care or a same-day appointment, rather than self-care.",
    "988": "The triage desk flagged thoughts of suicide or self-harm. Be warm and supportive, keep 988 in view, "
    "and let them lead.",
}


PAST_HINT = (
    "The person mentioned emergency warning signs, but seems to describe something that already happened. "
    "First confirm it is not happening now. Then treat it as history. Ask what happened and what was found, "
    "and suggest the right follow-up."
)


# In the scenario run the coordinator asked 6 or more questions in two cases
# and never reached an answer. The prompt already said three to five. So the
# count now comes from the state, and past the limit it is told to stop.
MAX_QUESTIONS = 5
ENOUGH_NOTE = (
    "You have already asked {n} questions. Do not ask another one. "
    "Consult the department that fits if you have not yet, then give your analysis now."
)


def questions_asked(messages) -> int:
    """How many questions the coordinator has asked since its last analysis."""
    count = 0
    for message in reversed(messages):
        if message.type != "ai" or message.tool_calls or (message.response_metadata or {}).get("sympai_triage"):
            continue
        text = message.content or ""
        if "Most likely" in text:
            break
        if text.rstrip().endswith("?"):
            count += 1
    return count


PEAK_HINTS = {
    "urgent": "Earlier in this conversation the triage desk flagged this as needing a doctor soon. "
    "Your answer must include same-day or next-day care, and a clear line on when to call 911.",
    "911": "Earlier in this conversation the triage desk flagged a possible emergency. "
    "Your answer must include a clear line on when to call 911, and same-day care at the least.",
}


CHART_NOTE = """What SympAI already knows about this person, from earlier conversations
{summary}
Use it, and do not ask again for anything listed here. If something may have changed, confirm it in passing.
This is the chart of the person using SympAI. If they are asking about someone else, it does not apply to that person."""


def system_prompt(state: Consult, chart_summary: str = "") -> str:
    """The coordinator's instructions for this call, built from the fixed prompt plus what the state says."""
    triage = state.get("triage") or {}
    hint = PAST_HINT if triage.get("past") else TRIAGE_HINTS.get(triage.get("level"))
    notes = [f"Triage desk\n{hint}"] if hint else []
    peak_hint = PEAK_HINTS.get(state.get("peak"))
    if peak_hint and peak_hint != hint:
        notes.append(peak_hint)
    asked = questions_asked(state.get("messages", []))
    if asked >= MAX_QUESTIONS:
        notes.append(ENOUGH_NOTE.format(n=asked))
    if chart_summary:
        notes.append(CHART_NOTE.format(summary=chart_summary))
    language = (state.get("language") or "").lower()
    if language and language != "english":
        notes.append(f"The person is speaking {language}. Reply in {language}, in short, simple sentences.")
    return "\n\n".join([COORDINATOR_PROMPT, *notes])


def _consult_department(department: str, case_summary: str) -> str:
    report, _ = consult(department, case_summary)
    return report.model_dump_json()


def consult_department_tool() -> StructuredTool:
    """consult_department as the coordinator sees it. The list of departments
    is read from the YAML files, so a new file shows up here on its own."""
    return StructuredTool.from_function(
        func=_consult_department,
        name="consult_department",
        description=(
            "Ask one hospital department for a specialist opinion. You get back a typed report "
            "with what is most likely, what else is possible, how sure, red flags, and what would confirm it. "
            "Pass the department key and a case_summary with age group, symptoms, when they started, "
            "how bad they are, and every answer the person gave.\n"
            "Departments\n" + departments_for_coordinator()
        ),
    )


def tool_rounds_this_turn(messages) -> int:
    """How many times the coordinator has asked for tools since the person last spoke."""
    rounds = 0
    for message in reversed(messages):
        if message.type == "human":
            break
        if message.type == "ai" and message.tool_calls:
            rounds += 1
    return rounds


def context_for_model(messages):
    """What the model is sent, which is not the same as what the state keeps.

    Every call resends the whole conversation. In testing, one thread grew to
    6,126 tokens per call and hit Groq's 8,000-a-minute limit. Tool results
    from earlier turns have already been read and summed up in an answer, so
    here they are cut to a one-line stub. The full text stays in the state,
    and Studio still shows it. Choosing what goes into the window each call
    is called context engineering. Step 5 goes further with summaries.
    """
    last_human = max((i for i, m in enumerate(messages) if m.type == "human"), default=0)
    trimmed = []
    for i, message in enumerate(messages):
        if message.type == "ai" and (message.response_metadata or {}).get("sympai_busy"):
            continue  # a "come back later" note is for the person, not something the model should build on
        if message.type == "ai" and (message.response_metadata or {}).get("sympai_status"):
            # The "Asking the ENT department..." line is for the person. The model gets its message back as it wrote it.
            message = message.model_copy(update={"content": ""})
        if message.type == "human" and not isinstance(message.content, str):
            # Content blocks from a chat screen become plain text. The models on
            # Groq cannot see images or read PDFs yet, so say so instead of failing.
            words = as_text(message.content)
            if has_attachment(message.content):
                words += ("\n\n(The person attached an image or file. SympAI cannot read attachments yet, "
                          "so ask them to type what it shows, such as the test names and numbers on a lab report.)")
            message = HumanMessage(words, id=message.id)
        if i < last_human and message.type == "tool":
            message = ToolMessage(
                content=f"({message.name} result from an earlier turn, removed to save space)",
                tool_call_id=message.tool_call_id,
                name=message.name,
            )
        trimmed.append(message)
    return trimmed


def check_output(reply: AIMessage, state: Consult) -> AIMessage:
    """The output guardrails, in code, on every reply the person will read."""
    cleaned = clean_answer(reply.content)
    reply.content = one_question(cleaned)
    if reply.content != cleaned:
        reply.response_metadata["sympai_trimmed_question"] = True
    reply.content, rerouted = route_serious_signs_to_911(reply.content)
    if rerouted:
        reply.response_metadata["sympai_rerouted"] = True
    reply.content, raised = enforce_minimum_care(reply.content, state.get("peak"))
    if raised:
        reply.response_metadata["sympai_raised_care"] = True
    return reply


def is_rate_limit(error: Exception) -> bool:
    """Groq answers 429 when a per-minute or per-day allowance is used up."""
    return getattr(error, "status_code", None) == 429 or "rate limit" in str(error).lower()


def is_bad_generation(error: Exception) -> bool:
    """Groq answers 400 when the model writes a broken lookup request it cannot read.

    In the visual snow test the 20b backup did this on "How do I go to the
    neurology department here?", and the whole turn crashed with no reply.
    Each generation is different, so a second try usually works.
    """
    return getattr(error, "status_code", None) == 400


def is_server_side(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    return isinstance(status, int) and status >= 500


def wait_seconds(error: Exception, default: float = 60.0) -> float:
    """Read Groq's "Please try again in 14m36.96s" out of the error."""
    match = re.search(r"try again in ((?:\d+h)?(?:\d+m)?(?:[\d.]+s)?)", str(error))
    if not match or not match.group(1):
        return default
    parts = {unit: float(number) for number, unit in re.findall(r"([\d.]+)([hms])", match.group(1))}
    return parts.get("h", 0) * 3600 + parts.get("m", 0) * 60 + parts.get("s", 0)


def say_wait(seconds: float) -> str:
    minutes = max(1, round(seconds / 60))
    return "a minute" if minutes == 1 else f"{minutes} minutes" if minutes < 90 else f"{round(minutes / 60)} hours"


class Breaker:
    """A tiny circuit breaker, the same idea as Resilience4j in Spring.

    Once Groq says the main model is out of tokens, every new call would spend
    about 25 seconds on retries before failing again. So the breaker opens for
    as long as Groq said to wait, and calls go straight to the backup until then.
    """

    def __init__(self, clock=time.time):
        self.clock = clock
        self.open_until = 0.0

    def is_open(self) -> bool:
        return self.clock() < self.open_until

    def trip(self, error: Exception) -> None:
        self.open_until = self.clock() + wait_seconds(error)

    def remaining(self) -> float:
        return max(0.0, self.open_until - self.clock())


def make_coordinator(brain, closer, chart_store=None, backup=None, backup_closer=None, breaker=None):
    """`brain` may call tools. `closer` is the same model with tools switched off.

    `backup` and `backup_closer` are the same pair on a second model, used
    only while the main one is rate limited.
    """
    breaker = breaker or Breaker()

    def think(state: Consult, history, prompt: str) -> AIMessage:
        closing = tool_rounds_this_turn(history) >= MAX_TOOL_ROUNDS
        with_tools = [SystemMessage(prompt), *history]
        without_tools = [SystemMessage(prompt + "\n\n" + (BUDGET_NOTE if closing else NO_TOOLS_NOTE)), *history]
        choices = [(brain, closer, False)]
        if backup is not None:
            choices.append((backup, backup_closer or backup, True))
        wait, went_wrong = 0.0, False
        for tools_model, text_model, is_backup in choices:
            if not is_backup and breaker.is_open():
                wait = breaker.remaining()
                continue
            # Normally the model may look things up, with one retry if its request
            # comes out broken, then a last try with lookups switched off.
            attempts = [(text_model, True)] if closing else [(tools_model, False), (tools_model, False), (text_model, True)]
            for model, text_only in attempts:
                try:
                    answer = model.invoke(without_tools if text_only else with_tools)
                except Exception as error:
                    if is_rate_limit(error):
                        if not is_backup:
                            breaker.trip(error)
                        wait = max(wait, wait_seconds(error))
                        break  # this model is out, go to the next one
                    if is_server_side(error):
                        log.warning("coordinator model failed on Groq's side: %s", error)
                        went_wrong = True
                        break
                    if is_bad_generation(error):
                        log.warning("coordinator wrote a request Groq could not read, trying again: %s", error)
                        went_wrong = True
                        continue
                    raise  # anything else is a bug in our code, and hiding it would make it harder to find
                flags = {"sympai_backup": True} if is_backup else {}
                if text_only:
                    # A fresh message with no tool calls, whatever the model tried, so
                    # the turn is guaranteed to end here. The flag is only for display.
                    flags["sympai_budget_closed" if closing else "sympai_no_tools"] = True
                    return AIMessage(
                        content=answer.content,
                        id=answer.id,
                        usage_metadata=answer.usage_metadata,
                        additional_kwargs=answer.additional_kwargs,
                        response_metadata={**answer.response_metadata, **flags},
                    )
                answer.response_metadata = {**answer.response_metadata, **flags}
                return answer
        if went_wrong:
            # Never silence. An honest "please send that again", with the safety lines.
            return AIMessage(content=SORRY_REPLY, response_metadata={"sympai_busy": True, "sympai_error": True})
        # Both models are out. An honest "not now", with the safety lines, beats a crash.
        return AIMessage(content=BUSY_REPLY.format(wait=say_wait(wait)), response_metadata={"sympai_busy": True})

    def coordinator(state: Consult, config: RunnableConfig) -> dict:
        # Long-term memory in, as a short summary. The chart file itself never leaves the laptop.
        chart = charts.load(chart_store, charts.person_of(config))
        prompt = system_prompt(state, charts.summary(chart))
        reply = think(state, context_for_model(state["messages"]), prompt)
        # gpt-oss returns its private reasoning with each reply, and the chat
        # screen shows it as "Reasoning". Useful for learning, noisy for
        # friends, so it is dropped unless SYMPAI_SHOW_REASONING=1 is set.
        if not os.getenv("SYMPAI_SHOW_REASONING"):
            reply.additional_kwargs.pop("reasoning_content", None)
        if reply.tool_calls and not reply.content:
            reply.content = status_line(reply.tool_calls)
            reply.response_metadata = {**reply.response_metadata, "sympai_status": True}
        if not reply.tool_calls and not reply.response_metadata.get("sympai_busy"):
            reply = check_output(reply, state)
            already_told = any(
                m.type == "ai" and (m.response_metadata or {}).get("sympai_backup_noted") for m in state["messages"]
            )
            if reply.response_metadata.get("sympai_backup") and not already_told:
                # Said once per conversation, because a person deserves to know a smaller model took over.
                reply.content = f"{reply.content}\n\n{BACKUP_NOTE}"
                reply.response_metadata["sympai_backup_noted"] = True
        return {"messages": [reply]}

    return coordinator


def department(state: Consult) -> dict:
    """Runs every consult_department call in the coordinator's last message."""
    last = state["messages"][-1]
    messages, reports = [], []
    for call in last.tool_calls:
        if call["name"] != "consult_department":
            continue
        key = str(call["args"].get("department", "")).strip().lower()
        if key not in load_departments():
            content = f"There is no department '{key}'. Choose one of {', '.join(load_departments())}."
        else:
            try:
                report, answered_by = consult(key, call["args"].get("case_summary", ""))
                # answered_by stays in our state for debugging. The coordinator
                # only gets the report itself.
                reports.append({**report.model_dump(), "answered_by": answered_by})
                content = report.model_dump_json()
            except Exception as error:  # both models failed, e.g. rate limits
                # One short line. The full error was about a thousand characters of
                # JSON, and it went to the model every call after, costing tokens.
                content = f"The {key} department could not answer this time ({type(error).__name__}). Continue without it and say so."
        messages.append(ToolMessage(content=content, tool_call_id=call["id"], name="consult_department"))
    return {"messages": messages, "reports": reports}


def make_remember(clerk, chart_store):
    """After each turn, the chart clerk folds anything new into the person's chart."""

    def remember(state: Consult, config: RunnableConfig) -> dict:
        if clerk is None or chart_store is None:
            return {}
        messages = state["messages"]
        last_human = next((m for m in reversed(messages) if m.type == "human"), None)
        if last_human is None or messages[-1].response_metadata.get("sympai_busy"):
            return {}  # nothing was advised, so there is nothing to record
        person = charts.person_of(config)
        try:
            updated = charts.update(clerk, charts.load(chart_store, person), last_human.content, messages[-1].content)
            charts.save(chart_store, person, updated)
        except Exception:
            pass  # memory is best effort. A failed update never breaks the conversation.
        return {}

    return remember


def research_tools() -> list:
    # Looked up each call, not at import, so tests can swap in fakes.
    return [search_medlineplus, search_pubmed, search_clinical_trials]


def service_tools() -> list:
    return [find_clinicians, care_options]


def make_tool_node(get_tools, answers_unknown: bool = False):
    """A node that runs the coordinator's calls to its own tools, and ignores the rest.

    The research node also answers calls to tools that do not exist, like the
    browser.search that gpt-oss sometimes invents, so no call goes unanswered.
    """

    def run_tools(state: Consult) -> dict:
        tools = {t.name: t for t in get_tools()}
        known = {"consult_department"} | {t.name for t in research_tools() + service_tools()}
        messages = []
        for call in state["messages"][-1].tool_calls:
            artifact = None
            if call["name"] in tools:
                try:
                    # Passing the whole call, not just its args, returns a ToolMessage,
                    # which carries the tool's artifact, such as clinician cards.
                    result = tools[call["name"]].invoke({**call, "type": "tool_call"})
                    content = result.content if isinstance(result, ToolMessage) else result
                    artifact = getattr(result, "artifact", None)
                except Exception as error:  # bad arguments from the model, most often
                    content = f"{call['name']} failed ({error}). Answer without it and say so."
            elif answers_unknown and call["name"] not in known:
                content = f"There is no tool called {call['name']}. Use only the tools you were given."
            else:
                continue
            messages.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"], artifact=artifact))
        return {"messages": messages}

    return run_tools


research = make_tool_node(research_tools, answers_unknown=True)
services = make_tool_node(service_tools)


def route(state: Consult):
    """The conditional edge after the coordinator.

    No tool calls means the coordinator wrote to the person, so the chart is
    updated, the turn ends, and we wait for their reply. Otherwise go to whichever nodes can answer
    the calls. Returning two nodes runs them in parallel, and the coordinator
    runs once after both finish.
    """
    last = state["messages"][-1]
    if not getattr(last, "tool_calls", None):
        return "remember"
    names = {call["name"] for call in last.tool_calls}
    service_names = {t.name for t in service_tools()}
    targets = []
    if "consult_department" in names:
        targets.append("department")
    if names & service_names:
        targets.append("services")
    if names - {"consult_department"} - service_names:
        targets.append("research")
    return targets


def build_graph(
    brain=None, closer=None, checkpointer=None, guard=None, classifier=None, chart_store=None, clerk=None,
    backup=None, backup_closer=None,
):
    """Wire the hospital together.

    `brain`, `closer` and `guard` can be passed in, which is dependency
    injection. Tests hand in fakes so the whole graph runs offline without
    spending tokens. With a fake brain, Prompt Guard and the classifier stay
    off unless a test passes them in.
    """
    if brain is None:
        guard = guard or prompt_guard_score
        classifier = classifier or classify
        chart_store = chart_store or charts.LazyStore()
        clerk = clerk or charts.default_clerk()
        base = ChatGroq(
            model=COORDINATOR_MODEL, temperature=0, reasoning_effort="medium", max_retries=MAX_RETRIES
        )
        tools = [consult_department_tool(), *research_tools(), *service_tools()]
        brain = base.bind_tools(tools)
        # Same model and tools, but tool_choice="none" means it may not call
        # any. The tools stay listed because the history already has tool calls in it.
        closer = base.bind_tools(tools, tool_choice="none")
        # Fewer retries on the backup. If it is also out, the person hears so quickly.
        spare = ChatGroq(model=BACKUP_COORDINATOR_MODEL, temperature=0, reasoning_effort="medium", max_retries=2)
        backup = spare.bind_tools(tools)
        backup_closer = spare.bind_tools(tools, tool_choice="none")
    closer = closer or brain

    builder = StateGraph(Consult)
    builder.add_node("triage", make_triage(guard, classifier))
    builder.add_node("coordinator", make_coordinator(brain, closer, chart_store, backup, backup_closer))
    builder.add_node("department", department)
    builder.add_node("research", research)
    builder.add_node("services", services)
    builder.add_node("remember", make_remember(clerk, chart_store))

    builder.add_edge(START, "triage")
    builder.add_conditional_edges("triage", after_triage, ["coordinator", END])
    builder.add_conditional_edges("coordinator", route, ["department", "research", "services", "remember"])
    builder.add_edge("remember", END)
    builder.add_edge("department", "coordinator")
    builder.add_edge("research", "coordinator")
    builder.add_edge("services", "coordinator")

    return builder.compile(checkpointer=checkpointer)
