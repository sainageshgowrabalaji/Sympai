"""Tests for the chart, SympAI's long-term memory. Fake models, a real in-memory Store."""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from sympai import chart as charts
from sympai.chart import EMPTY, Chart
from sympai.graph import build_graph


class Coordinator:
    """A fake coordinator that remembers the system prompt it was given."""

    def __init__(self):
        self.prompts = []

    def __call__(self, messages):
        self.prompts.append(messages[0].content)
        return AIMessage("When did it start?")


def clerk_that_learns(**facts):
    return RunnableLambda(lambda messages: EMPTY.model_copy(update=facts))


def talk(graph, text, thread, person):
    config = {"configurable": {"thread_id": thread, "person": person}}
    return graph.invoke({"messages": [HumanMessage(text)]}, config)


def test_every_chart_field_is_required_for_strict_mode():
    schema = Chart.model_json_schema()
    assert set(schema["required"]) == set(schema["properties"])


def test_an_empty_chart_adds_nothing_to_the_prompt():
    assert charts.summary(EMPTY) == ""


def test_the_chart_carries_over_to_a_new_conversation():
    store, coordinator = InMemoryStore(), Coordinator()
    graph = build_graph(
        brain=RunnableLambda(coordinator), checkpointer=InMemorySaver(), chart_store=store,
        clerk=clerk_that_learns(age="34", insurance="Aetna Open Access", has_primary_care=False),
    )
    talk(graph, "I'm 34, I have Aetna Open Access and no primary care doctor", thread="monday", person="alex")
    assert "Age 34" not in coordinator.prompts[0]  # nothing known yet on the very first call

    talk(graph, "my ears are ringing again", thread="friday", person="alex")  # a brand new thread
    assert "Age 34" in coordinator.prompts[1]
    assert "Insurance Aetna Open Access" in coordinator.prompts[1]
    assert "Has no primary care doctor yet" in coordinator.prompts[1]


def test_one_persons_chart_is_never_shown_to_another():
    store, coordinator = InMemoryStore(), Coordinator()
    graph = build_graph(
        brain=RunnableLambda(coordinator), checkpointer=InMemorySaver(), chart_store=store,
        clerk=clerk_that_learns(age="34"),
    )
    talk(graph, "I'm 34", thread="a", person="alex")
    talk(graph, "hello", thread="b", person="friend")
    assert "Age 34" not in coordinator.prompts[1]


def test_a_failing_clerk_never_breaks_the_conversation():
    def broken(messages):
        raise RuntimeError("rate limit")

    graph = build_graph(
        brain=RunnableLambda(Coordinator()), checkpointer=InMemorySaver(), chart_store=InMemoryStore(),
        clerk=RunnableLambda(broken),
    )
    result = talk(graph, "my ears ring", thread="t", person="alex")
    assert result["messages"][-1].content == "When did it start?"


def test_forget_removes_the_chart():
    store = InMemoryStore()
    charts.save(store, "alex", EMPTY.model_copy(update={"age": "34"}))
    charts.forget(store, "alex")
    assert charts.load(store, "alex") == EMPTY


def test_the_sqlite_store_keeps_charts_across_restarts(tmp_path):
    path = tmp_path / "charts.sqlite"
    charts.save(charts.open_chart_store(path), "alex", EMPTY.model_copy(update={"zip_code": "07102"}))
    assert charts.load(charts.open_chart_store(path), "alex").zip_code == "07102"
