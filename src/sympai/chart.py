"""
Step 5. The chart. Long-term memory, one per person.

Thread memory (session.py) remembers one conversation. The chart remembers
the person across all of them, so next week SympAI already knows their age,
their insurance and what it advised last time, instead of asking again.

Two jobs, two moments
    read    at the start of every coordinator call, as a short summary in its prompt
    write   at the end of every turn, by a small "chart clerk" model

The chart lives in LangGraph's Store, a key-value store with namespaces, like
a table keyed by (namespace, key). Each person gets the namespace
("charts", person). The Store is backed by data/charts.sqlite on your laptop,
one file shared by the terminal chat, the chat screen and scripts/chart.py.
Only the short summary ever travels to Groq, never the file.
"""

import sqlite3
from datetime import date
from functools import cache
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.store.sqlite import SqliteStore
from pydantic import BaseModel, Field

from sympai.text import as_text

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CHART_DB = DATA_DIR / "charts.sqlite"
CLERK_MODEL = "openai/gpt-oss-20b"  # small and fast, and its own free allowance
DEFAULT_PERSON = "me"


class Chart(BaseModel):
    """What SympAI keeps about one person. Every field is required so Groq's strict mode works."""

    age: str | None = Field(description="Age or age group, as they said it")
    conditions: list[str] = Field(description="Long-term conditions they said they have")
    medicines: list[str] = Field(description="Medicines they said they take")
    allergies: list[str] = Field(description="Allergies they mentioned")
    insurance: str | None = Field(description="Insurance plan name, if they said it")
    zip_code: str | None = Field(description="Their ZIP code, if they gave it")
    has_primary_care: bool | None = Field(description="Whether they have a primary care doctor, if known")
    history: list[str] = Field(description="Past and current health episodes, in their words, with when")
    advice: list[str] = Field(description="What SympAI advised, one short line each, starting with the date")


EMPTY = Chart(
    age=None, conditions=[], medicines=[], allergies=[], insurance=None, zip_code=None,
    has_primary_care=None, history=[], advice=[],
)

CLERK_PROMPT = """You keep the chart for SympAI, a health navigation app. You get the current chart and the latest exchange. Return the updated chart.

Rules
- Record only what the person stated about themselves. If they are asking about someone else, such as a parent or friend, record nothing about that other person.
- Keep everything already in the chart unless the person corrects it.
- Never guess or infer a condition. A symptom goes in history in their words, with when it happened.
- Add a line to advice only if SympAI gave an analysis or a recommendation in this exchange. Start it with {today}.
- Keep each list to 10 short items at most. Merge duplicates."""


def open_chart_store(path: Path | None = None) -> SqliteStore:
    path = path or CHART_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    # isolation_level=None is autocommit mode, which the store expects because it
    # starts its own transactions. The library's own from_conn_string does the
    # same. Without it the second write failed with "transaction within a transaction".
    store = SqliteStore(sqlite3.connect(path, check_same_thread=False, isolation_level=None))
    store.setup()  # creates the tables the first time, like a schema migration
    return store


class LazyStore:
    """Opens data/charts.sqlite on first use instead of at import.

    The LangGraph server imports the graph while starting up, and a database
    opened there would be blocking work on its event loop. This defers it to
    the first chart read, which happens inside a node, on a worker thread.
    """

    def __init__(self, path: Path | None = None):
        self._path, self._store = path, None

    def _open(self) -> SqliteStore:
        if self._store is None:
            self._store = open_chart_store(self._path)
        return self._store

    def get(self, *args, **kwargs):
        return self._open().get(*args, **kwargs)

    def put(self, *args, **kwargs):
        return self._open().put(*args, **kwargs)

    def delete(self, *args, **kwargs):
        return self._open().delete(*args, **kwargs)

    def search(self, *args, **kwargs):
        return self._open().search(*args, **kwargs)


def namespace(person: str) -> tuple[str, str]:
    return ("charts", person)


def person_of(config: dict | None) -> str:
    """Whose chart this conversation belongs to.

    Through the server, the signed-in name always wins. The server sets
    langgraph_auth_user_id itself after checking the key, while "person" is
    something any caller can type into a request. If "person" came first, a
    friend could send person="me" and read your chart. The terminal chat has
    no sign-in, so there it passes "person".
    """
    configurable = (config or {}).get("configurable", {})
    return configurable.get("langgraph_auth_user_id") or configurable.get("person") or DEFAULT_PERSON


def load(store, person: str) -> Chart:
    item = store.get(namespace(person), "chart") if store else None
    return Chart(**item.value) if item else EMPTY


def save(store, person: str, chart: Chart) -> None:
    store.put(namespace(person), "chart", chart.model_dump())


def forget(store, person: str) -> None:
    store.delete(namespace(person), "chart")


def summary(chart: Chart) -> str:
    """The short version the coordinator sees. Empty when nothing is known yet."""
    lines = []
    if chart.age:
        lines.append(f"Age {chart.age}")
    for label, items in (("Conditions", chart.conditions), ("Medicines", chart.medicines), ("Allergies", chart.allergies)):
        if items:
            lines.append(f"{label} {', '.join(items)}")
    if chart.insurance:
        lines.append(f"Insurance {chart.insurance}")
    if chart.zip_code:
        lines.append(f"ZIP {chart.zip_code}")
    if chart.has_primary_care is not None:
        lines.append("Has a primary care doctor" if chart.has_primary_care else "Has no primary care doctor yet")
    if chart.history:
        lines.append("History " + "; ".join(chart.history[-5:]))
    if chart.advice:
        lines.append("Advised before " + "; ".join(chart.advice[-3:]))
    return "\n".join(f"- {line}" for line in lines)


@cache
def default_clerk():
    return ChatGroq(model=CLERK_MODEL, temperature=0, reasoning_effort="low", max_retries=6).with_structured_output(
        Chart, method="json_schema", strict=True
    )


def update(clerk, chart: Chart, person_said: str, sympai_said: str) -> Chart:
    """Ask the clerk for the updated chart after one exchange."""
    return clerk.invoke(
        [
            SystemMessage(CLERK_PROMPT.format(today=date.today().isoformat())),
            HumanMessage(
                f"Current chart\n{chart.model_dump_json()}\n\n"
                f"The person said\n{as_text(person_said)}\n\nSympAI said\n{as_text(sympai_said)}"
            ),
        ]
    )
