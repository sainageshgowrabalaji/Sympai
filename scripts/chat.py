"""
Talk to the hospital. It remembers, pauses and picks back up.

Run it
    uv run python scripts/chat.py

If you stop replying for 2 minutes, or press Ctrl+C, it pauses. The whole
conversation is already saved in data/conversations.sqlite on your laptop.
Next time you run it, it tells you where you were and carries on from there.

Try the timer quickly with a shorter wait
    SYMPAI_IDLE_SECONDS=15 uv run python scripts/chat.py

Each person has their own chart and their own last conversation
    uv run python scripts/chat.py --person mom

Talk instead of typing, in any language, and hear the replies
    uv run python scripts/chat.py --voice --speak
"""

import argparse
import os
import select
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError

from sympai import chart as charts
from sympai import classifier, voice
from sympai.graph import build_graph
from sympai.session import (
    last_thread,
    new_thread_id,
    open_checkpointer,
    recap,
    remember_thread,
    thread_config,
    time_ago,
)

IDLE_SECONDS = int(os.getenv("SYMPAI_IDLE_SECONDS", "120"))


def timed_input(prompt: str, seconds: int = IDLE_SECONDS) -> str | None:
    """Like input(), but gives up after `seconds` of silence and returns None.

    select() waits until the keyboard has something to read or the time runs
    out, whichever comes first. It works in the macOS and Linux terminal.
    """
    print(prompt, end="", flush=True)
    ready, _, _ = select.select([sys.stdin], [], [], seconds)
    if not ready:
        return None
    line = sys.stdin.readline()
    if not line:  # Ctrl+D
        raise EOFError
    return line.strip()


def wait_words(seconds: int) -> str:
    return f"{seconds // 60} minutes" if seconds >= 120 else f"{seconds} seconds"


def short(text: str, limit: int = 160) -> str:
    one_line = " ".join(text.replace("**", "").split())
    return one_line if len(one_line) <= limit else one_line[:limit].rsplit(" ", 1)[0] + " ..."


def pause(reason: str = "") -> None:
    print(f"\n\n({reason}Paused. Everything so far is saved on your laptop. Run chat.py again to pick up right here.)")


def _names(findings: list[dict]) -> str:
    return ", ".join(f"{f['name']} ({f['confidence']})" for f in findings) or "none"


def show(node: str, update: dict) -> None:
    """Print what each node just did, so you can watch the hospital work."""
    if node == "triage":
        result = update["triage"]
        score = result["injection_score"]
        extra = f", prompt check {score:.2f}" if score is not None else ""
        what = ", ".join(result["rules"]) or "prompt attack"
        if result["level"] == "clear":
            print(f"   . triage desk clear{extra}")
        elif result["blocking"]:
            print(f"   . TRIAGE DESK {result['level'].upper()} ({what}), the turn ends here")
        elif result["shown"]:
            print(f"   . TRIAGE DESK {result['level'].upper()} possible ({what}), warned, the coordinator carries on")
        else:
            print(f"   . triage desk {result['level']} ({what}), no new notice, the coordinator is told")
        for message in update.get("messages", []):
            print(f"\nSympAI: {message.content}")
    elif node == "coordinator":
        reply = update["messages"][-1]
        tokens = (reply.usage_metadata or {}).get("total_tokens")
        for call in reply.tool_calls:
            if call["name"] == "consult_department":
                print(f"   . coordinator asks the {str(call['args'].get('department')).upper()} department")
                # What the department actually receives. When a report looks
                # wrong, check this first. A thin summary gets a thin report.
                print(f"       case sent  {call['args'].get('case_summary', '')}")
            else:
                print(f"   . coordinator calls {call['name']}({call['args']})")
        if reply.response_metadata.get("sympai_raised_care"):
            print("   . code added same-day care or a 911 line, because triage flagged this conversation")
        if reply.response_metadata.get("sympai_rerouted"):
            print("   . code moved a serious warning sign from urgent care to 911")
        if reply.response_metadata.get("sympai_trimmed_question"):
            print("   . it asked two questions at once, so code kept only the first")
        if reply.response_metadata.get("sympai_budget_closed"):
            print("   . lookup budget used up, so it answers with what it has")
        if reply.response_metadata.get("sympai_backup"):
            print("   . the 120b model is out of its free allowance, so the 20b backup answered")
        if reply.response_metadata.get("sympai_busy"):
            print("   . both models are out of their free allowance for now")
        print(f"   . coordinator used {tokens} tokens")
        if not reply.tool_calls:
            print(f"\nSympAI: {reply.content}")
    elif node == "department":
        for report in update.get("reports", []):
            scope = "in scope" if report["in_scope"] else "OUT of scope"
            print(f"   . {report['department']} reports back, {scope}, written by {report['answered_by']}")
            print(f"       most likely    {_names(report['most_likely'])}")
            print(f"       also possible  {_names(report['also_possible'])}")
            if report["what_would_confirm"]:
                print(f"       would confirm  {'; '.join(report['what_would_confirm'])}")
            if report["refer_to"]:
                print(f"       refers to      {'; '.join(report['refer_to'])}")
            for flag in report["red_flags_seen"]:
                print(f"       RED FLAG       {flag}")
        for message in update.get("messages", []):
            if not message.content.startswith("{"):
                print(f"   . {message.content}")
    elif node == "research":
        for message in update["messages"]:
            print(f"   . {message.name} returned {len(message.content)} characters")


def run_turn(graph, payload, config) -> None:
    """One turn. payload=None means finish a turn that was cut off earlier."""
    try:
        for update in graph.stream(payload, config, stream_mode="updates"):
            for node, change in update.items():
                show(node, change)
    except GraphRecursionError:
        print("\n(Stopped. The agents went around too many times without answering.)")
    except Exception as error:  # a rate limit or a network problem, most often
        print(f"\n(The model service failed with {type(error).__name__}: {error})")
        print("(Your message is saved. Press Enter on an empty line to try again.)")


def choose_thread(graph, person: str) -> tuple[str | None, str | None]:
    """Offer to carry on the last conversation.

    Returns the thread and, if the person answered straight away instead of
    pressing Enter, what they typed. (None, None) means they went quiet.
    """
    thread_id = last_thread(person)
    if thread_id:
        config = thread_config(thread_id, person)
        where = recap(graph.get_state(config))
        if where:
            print(f"\nWelcome back. {time_ago(where['when'])} we were talking about \"{short(where['topic'])}\".")
            if where["unfinished"]:
                print("I had not finished answering your last message.")
            elif where["last_said"].rstrip().endswith("?"):
                print(f"I had just asked you \"{short(where['last_said'])}\"")
            elif where["last_said"]:
                print(f"The last thing I said was \"{short(where['last_said'])}\"")
            answer = timed_input("Press Enter to carry on, type new to start fresh, or just reply: ")
            if answer is None:
                return None, None
            if answer.lower() != "new":
                if where["unfinished"]:
                    run_turn(graph, None, config)
                # In testing you typed your answer right here, and it was thrown away.
                return thread_id, answer or None
    thread_id = new_thread_id()
    remember_thread(thread_id, person)
    return thread_id, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Talk to SympAI.")
    parser.add_argument("--person", default=charts.DEFAULT_PERSON, help="whose chart to use, like me or mom")
    parser.add_argument("--voice", action="store_true", help="press Enter on an empty line to talk")
    parser.add_argument("--speak", action="store_true", help="read SympAI's replies aloud")
    options = parser.parse_args()
    person = options.person
    load_dotenv()
    graph = build_graph(checkpointer=open_checkpointer())
    if classifier.MODEL_DIR.exists():
        print("Loading the triage classifier...")
        if not classifier.available():
            print("(The saved classifier is out of date. Run scripts/train_triage.py again. Using rules only.)")

    print(f"SympAI. Talking with {person}. Tell it what is going on.")
    print(f"If you do not reply for {wait_words(IDLE_SECONDS)}, it pauses and saves.")

    try:
        known = charts.summary(charts.load(charts.open_chart_store(), person))
        if known:
            print(f"\nWhat SympAI remembers about {person}\n{known}")
        thread_id, first_reply = choose_thread(graph, person)
        if thread_id is None:
            return pause("No reply. ")
        config = thread_config(thread_id, person)
        if first_reply:
            run_turn(graph, {"messages": [HumanMessage(first_reply)]}, config)

        prompt = "\nYou (press Enter to talk, or type): " if options.voice else "\nYou: "
        while True:
            text = timed_input(prompt)
            if text is None:
                return pause(f"No reply for {wait_words(IDLE_SECONDS)}. ")
            if text.lower() in {"quit", "exit"}:
                return pause()
            if not text and graph.get_state(config).next:  # an earlier turn failed, retry it
                run_turn(graph, None, config)
                continue
            payload = {"messages": [HumanMessage(text)]}
            if not text:
                if not options.voice:
                    continue
                words, language = voice.transcribe(voice.listen())
                if not words:
                    print("   . I did not catch that. Press Enter and try again.")
                    continue
                print(f"You said ({language}): {words}")
                payload = {"messages": [HumanMessage(words)], "language": language}
            run_turn(graph, payload, config)
            if options.speak:
                last = graph.get_state(config).values["messages"][-1]
                if last.type == "ai" and last.content:
                    voice.speak(last.content, graph.get_state(config).values.get("language") or "english")
    except (KeyboardInterrupt, EOFError):
        # Even mid-answer this is safe. The last finished step is saved, and
        # next time it offers to finish the answer.
        pause()


if __name__ == "__main__":
    main()
