"""
Pausing and picking a conversation back up.

The checkpointer saves the graph's state after every step. In step 3 it was
InMemorySaver, so everything vanished when the script ended. SqliteSaver
writes the same checkpoints to a file on your laptop instead. A conversation
now survives a phone call, a closed terminal or a restart, and carrying on is
just running the graph again with the same thread_id. In Java terms, the
session moved from the JVM heap to a database.

It also gives us durable execution. If a turn fails halfway, for example on
a rate limit, the last good step is already saved, and graph.stream(None, ...)
runs only what is left.

The file holds real conversations, so it lives in data/, which git ignores.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from sympai.text import as_text

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DB_PATH = DATA_DIR / "conversations.sqlite"
LAST_THREAD_FILE = DATA_DIR / "last_thread.json"


def open_checkpointer(db_path: Path = DB_PATH) -> SqliteSaver:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False because LangGraph runs parallel nodes (department
    # and research) on worker threads. SqliteSaver has its own lock.
    return SqliteSaver(sqlite3.connect(db_path, check_same_thread=False))


def new_thread_id() -> str:
    return str(uuid.uuid4())


def _read_last_threads(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    # The first version stored one thread for everyone, as {"thread_id": ...}.
    return {"me": data["thread_id"]} if "thread_id" in data else data


def remember_thread(thread_id: str, person: str = "me", path: Path = LAST_THREAD_FILE) -> None:
    """Each person has their own last conversation, so resuming never opens someone else's."""
    path.parent.mkdir(parents=True, exist_ok=True)
    threads = _read_last_threads(path)
    threads[person] = thread_id
    path.write_text(json.dumps(threads))


def last_thread(person: str = "me", path: Path = LAST_THREAD_FILE) -> str | None:
    return _read_last_threads(path).get(person)


def thread_config(thread_id: str, person: str = "me") -> dict:
    # recursion_limit caps the steps in one turn, the same guard as MAX_STEPS
    # in step 2. The lookup budget in graph.py ends every turn well before it.
    # person says whose chart this conversation reads and updates.
    return {"configurable": {"thread_id": thread_id, "person": person}, "recursion_limit": 12}


def recap(snapshot) -> dict | None:
    """Enough to say where we were, read from the saved state, for zero tokens."""
    messages = snapshot.values.get("messages", []) if snapshot.values else []
    if not messages:
        return None
    topic = next((as_text(m.content) for m in messages if m.type == "human"), "")
    last_said = next(
        (m.content for m in reversed(messages) if m.type == "ai" and not m.tool_calls and m.content),
        "",
    )
    return {
        "topic": topic,
        "last_said": last_said,
        "when": snapshot.created_at,
        "unfinished": bool(snapshot.next),  # a turn that failed or was cut off
    }


def time_ago(iso_timestamp: str, now: datetime | None = None) -> str:
    then = datetime.fromisoformat(iso_timestamp)
    seconds = ((now or datetime.now(timezone.utc)) - then).total_seconds()
    minutes, hours = seconds / 60, seconds / 3600
    if minutes < 2:
        return "A minute ago"
    if minutes < 90:
        return f"{round(minutes)} minutes ago"
    if hours < 36:
        return f"{round(hours)} hours ago"
    return f"{round(hours / 24)} days ago"
