"""
Tests for the graph wiring. A fake model stands in for Groq, and the
department and MedlinePlus are replaced with stubs, so a whole consult runs
offline in a fraction of a second. Same idea as Mockito in a JUnit test.
"""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END

from sympai.departments import DepartmentReport, Finding
from sympai.graph import MAX_TOOL_ROUNDS, build_graph, context_for_model, route


def call(name, args=None, call_id=None):
    return {"name": name, "args": args or {}, "id": call_id or f"id-{name}", "type": "tool_call"}


def state_after(message):
    return {"messages": [HumanMessage("hi"), message], "reports": []}


def test_a_plain_reply_updates_the_chart_then_ends_the_turn():
    assert route(state_after(AIMessage("Do you have a cough?"))) == "remember"


def test_a_department_call_goes_to_the_department():
    assert route(state_after(AIMessage("", tool_calls=[call("consult_department")]))) == ["department"]


def test_a_search_goes_to_research():
    assert route(state_after(AIMessage("", tool_calls=[call("search_medlineplus")]))) == ["research"]


def test_both_at_once_run_side_by_side():
    both = AIMessage("", tool_calls=[call("consult_department"), call("search_medlineplus")])
    assert route(state_after(both)) == ["department", "research"]


class FakeSearch:
    name = "search_medlineplus"

    def invoke(self, args):
        return "Strep Throat\nhttps://medlineplus.gov/strepthroat.html\nA sore throat caused by bacteria."


REPORT = DepartmentReport(
    department="ENT",
    in_scope=True,
    most_likely=[Finding(name="Strep throat", why="Fever and sore throat without a cough", confidence="medium")],
    also_possible=[Finding(name="Viral sore throat", why="Most sore throats are viral", confidence="medium")],
    red_flags_seen=[],
    what_would_confirm=["rapid strep test"],
    refer_to=[],
    sources=[],
)


def test_a_whole_consult_across_two_turns(monkeypatch):
    monkeypatch.setattr("sympai.graph.consult", lambda department, case_summary: (REPORT, "fake-model"))
    monkeypatch.setattr("sympai.graph.search_medlineplus", FakeSearch())

    # What the "model" says, in order. Turn 1 is a question. Turn 2 asks the
    # department and MedlinePlus at once, then writes the answer.
    brain = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage("Do you have a cough?"),
                AIMessage(
                    "",
                    tool_calls=[
                        call("consult_department", {"department": "ENT", "case_summary": "adult, fever, sore throat, no cough"}),
                        call("search_medlineplus", {"query": "strep throat"}),
                    ],
                ),
                AIMessage("**Most likely** strep throat【1†L1-L4】"),
            ]
        )
    )
    graph = build_graph(brain=brain, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "test"}}

    first = graph.invoke({"messages": [HumanMessage("Sore throat and fever since yesterday")]}, config)
    assert first["messages"][-1].content == "Do you have a cough?"

    second = graph.invoke({"messages": [HumanMessage("No cough")]}, config)
    kinds = [m.type for m in second["messages"]]
    # The checkpointer kept turn 1, both tools answered, and the coordinator
    # ran once after them, not twice.
    assert kinds == ["human", "ai", "human", "ai", "tool", "tool", "ai"]
    assert second["reports"][0]["most_likely"][0]["name"] == "Strep throat"
    assert second["reports"][0]["answered_by"] == "fake-model"
    assert second["messages"][-1].content == "**Most likely** strep throat"


def test_the_lookup_budget_forces_an_answer(monkeypatch):
    """The visual snow bug. A model that keeps asking for tools must still
    answer once the budget is used up, instead of hitting the recursion limit."""
    monkeypatch.setattr("sympai.graph.search_medlineplus", FakeSearch())
    keeps_searching = GenericFakeChatModel(
        messages=iter(
            [AIMessage("", tool_calls=[call("search_medlineplus", {"query": f"try {i}"}, f"c{i}")]) for i in range(5)]
        )
    )
    closer = GenericFakeChatModel(messages=iter([AIMessage("MedlinePlus has no page on this, so here is my read.")]))
    graph = build_graph(brain=keeps_searching, closer=closer, checkpointer=InMemorySaver())

    result = graph.invoke({"messages": [HumanMessage("visual snow")]}, {"configurable": {"thread_id": "budget"}})

    last = result["messages"][-1]
    assert last.content == "MedlinePlus has no page on this, so here is my read."
    assert last.response_metadata["sympai_budget_closed"] is True
    assert sum(m.type == "tool" for m in result["messages"]) == MAX_TOOL_ROUNDS


def test_old_tool_results_are_trimmed_but_this_turns_are_kept():
    history = [
        HumanMessage("tinnitus"),
        AIMessage("", tool_calls=[call("search_medlineplus", {"query": "tinnitus"}, "old")]),
        ToolMessage("a very long tinnitus page " * 100, tool_call_id="old", name="search_medlineplus"),
        AIMessage("Here is what I found."),
        HumanMessage("and migraine?"),
        AIMessage("", tool_calls=[call("search_medlineplus", {"query": "migraine"}, "new")]),
        ToolMessage("the migraine page", tool_call_id="new", name="search_medlineplus"),
    ]
    sent = context_for_model(history)
    assert "removed to save space" in sent[2].content
    assert sent[2].tool_call_id == "old"  # still paired with its request
    assert sent[6].content == "the migraine page"
    assert len(history[2].content) > 1000  # the state itself is untouched


def test_the_real_graph_builds(monkeypatch, tmp_path):
    """A smoke test of the production wiring, with no fakes.

    Every other test passes a fake brain, so none of them ever ran the code
    path your Mac runs. That is how a missing function reached you as a crash.
    Building the real graph needs no network, only a key-shaped string.
    """
    monkeypatch.setenv("GROQ_API_KEY", "not-a-real-key")
    monkeypatch.setattr("sympai.chart.CHART_DB", tmp_path / "charts.sqlite")  # never touch the real charts
    graph = build_graph()
    assert set(graph.get_graph().nodes) >= {"triage", "coordinator", "department", "research", "services", "remember"}


def test_after_five_questions_the_coordinator_is_told_to_answer():
    from sympai.graph import questions_asked, system_prompt

    history = [HumanMessage("my chest feels heavy sometimes")]
    for question in ["When did it start?", "How long does it last?", "Any shortness of breath?",
                     "Does it spread to your arm?", "Does posture change it?"]:
        history += [AIMessage(question), HumanMessage("no")]
    assert questions_asked(history) == 5
    assert "Do not ask another one" in system_prompt({"messages": history, "triage": {}})


def test_an_analysis_resets_the_question_count():
    from sympai.graph import questions_asked

    history = [AIMessage("When did it start?"), AIMessage("**Most likely** a strain."), AIMessage("Any swelling?")]
    assert questions_asked(history) == 1



def test_a_chat_screen_message_in_content_blocks_works():
    """Agent Chat UI sends a list of content blocks, not a string. That crashed triage."""
    seen = []

    def brain(messages):
        seen.append(messages[-1].content)
        return AIMessage("When did the vertigo start?")

    graph = build_graph(brain=RunnableLambda(brain), checkpointer=InMemorySaver())
    blocks = [{"type": "text", "text": "I have vertigo and tinnitus"}, {"type": "image", "url": "https://x/y.png"}]
    result = graph.invoke({"messages": [HumanMessage(blocks)]}, {"configurable": {"thread_id": "ui"}})
    assert result["triage"]["level"] == "clear"
    assert result["messages"][-1].content == "When did the vertigo start?"
    assert seen[0].startswith("I have vertigo and tinnitus")  # the model got plain text
    assert "cannot read attachments yet" in seen[0]


def test_a_content_block_emergency_is_still_caught():
    graph = build_graph(brain=RunnableLambda(lambda m: AIMessage("x")), checkpointer=InMemorySaver())
    blocks = [{"type": "text", "text": "my face is drooping"}]
    result = graph.invoke({"messages": [HumanMessage(blocks)]}, {"configurable": {"thread_id": "ui2"}})
    assert result["triage"]["blocking"] is True



def test_private_reasoning_is_not_shown_to_the_person(monkeypatch):
    monkeypatch.delenv("SYMPAI_SHOW_REASONING", raising=False)
    thinking = AIMessage("When did it start?", additional_kwargs={"reasoning_content": "We need to follow guidelines..."})
    graph = build_graph(brain=RunnableLambda(lambda m: thinking), checkpointer=InMemorySaver())
    result = graph.invoke({"messages": [HumanMessage("my ears ring")]}, {"configurable": {"thread_id": "r"}})
    assert "reasoning_content" not in result["messages"][-1].additional_kwargs



def test_services_calls_go_to_the_services_node():
    msg = AIMessage("", tool_calls=[call("find_clinicians", {"specialty": "ENT", "zip_code": "07102"})])
    assert route(state_after(msg)) == ["services"]


def test_research_services_and_a_department_run_side_by_side():
    msg = AIMessage("", tool_calls=[call("consult_department"), call("search_pubmed"), call("care_options")])
    assert route(state_after(msg)) == ["department", "services", "research"]


def test_a_made_up_tool_still_gets_an_answer():
    from sympai.graph import research

    msg = AIMessage("", tool_calls=[call("browser.search", {"q": "x"}, "b1")])
    result = research({"messages": [HumanMessage("hi"), msg]})
    assert "There is no tool called browser.search" in result["messages"][0].content


class OutOfTokens(Exception):
    """Shaped like Groq's 429, without needing a network."""

    status_code = 429

    def __init__(self):
        super().__init__("Rate limit reached on tokens per day (TPD). Please try again in 14m36.96s.")


def out_of_tokens(_messages):
    raise OutOfTokens()


def test_groq_wait_time_is_read_from_the_error():
    from sympai.graph import say_wait, wait_seconds

    assert round(wait_seconds(OutOfTokens())) == 877
    assert wait_seconds(Exception("Please try again in 2.5s")) == 2.5
    assert wait_seconds(Exception("Please try again in 1h2m3s")) == 3723
    assert wait_seconds(Exception("something else")) == 60
    assert say_wait(877) == "15 minutes"


def test_the_backup_answers_when_the_main_model_is_out_of_tokens():
    backup = GenericFakeChatModel(messages=iter([AIMessage("When did it start?"), AIMessage("Is it worse at night?")]))
    main_calls = []

    def main(messages):
        main_calls.append(1)
        raise OutOfTokens()

    graph = build_graph(brain=RunnableLambda(main), backup=backup, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "backup"}}

    first = graph.invoke({"messages": [HumanMessage("my ear rings")]}, config)
    last = first["messages"][-1]
    assert last.content.startswith("When did it start?")
    assert "backup model answered" in last.content  # the person is told
    assert last.response_metadata["sympai_backup"] is True

    # The breaker is open now, so the main model is not tried again, and no time goes on retries.
    graph.invoke({"messages": [HumanMessage("last week")]}, config)
    assert len(main_calls) == 1


def test_when_both_models_are_out_the_person_gets_safety_lines_not_a_crash():
    graph = build_graph(brain=RunnableLambda(out_of_tokens), backup=RunnableLambda(out_of_tokens), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "busy"}}

    result = graph.invoke({"messages": [HumanMessage("my ear rings")]}, config)
    reply = result["messages"][-1]
    assert reply.response_metadata["sympai_busy"] is True
    assert "15 minutes" in reply.content and "911" in reply.content and "988" in reply.content
    # The note is not sent back to the model next time.
    assert all(not m.response_metadata.get("sympai_busy") for m in context_for_model(result["messages"]) if m.type == "ai")


def test_other_errors_are_not_hidden_behind_the_backup():
    import pytest

    def broken(_messages):
        raise ValueError("a real bug")

    backup = GenericFakeChatModel(messages=iter([AIMessage("should not be used")]))
    graph = build_graph(brain=RunnableLambda(broken), backup=backup, checkpointer=InMemorySaver())
    with pytest.raises(ValueError):
        graph.invoke({"messages": [HumanMessage("hi")]}, {"configurable": {"thread_id": "bug"}})


def test_a_lookup_shows_what_sympai_is_doing_instead_of_an_empty_bubble(monkeypatch):
    from sympai.graph import status_line

    monkeypatch.setattr("sympai.graph.consult", lambda department, case_summary: (REPORT, "fake-model"))
    monkeypatch.setattr("sympai.graph.search_medlineplus", FakeSearch())
    brain = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage("", tool_calls=[call("consult_department", {"department": "bones_muscles"}), call("search_medlineplus", {"query": "calf cramps"})]),
                AIMessage("**Most likely** a muscle cramp."),
            ]
        )
    )
    graph = build_graph(brain=brain, checkpointer=InMemorySaver())
    result = graph.invoke({"messages": [HumanMessage("calf cramps")]}, {"configurable": {"thread_id": "status"}})

    working = result["messages"][1]
    assert working.content == "_Asking the bones and muscles department and looking this up on MedlinePlus..._"
    # The model is sent its own message back empty, exactly as it wrote it.
    assert context_for_model(result["messages"])[1].content == ""
    assert status_line([call("find_clinicians", {"zip_code": "07102"})]) == "_Looking up clinicians near 07102..._"


def test_the_backup_note_is_said_once_per_conversation():
    backup = GenericFakeChatModel(messages=iter([AIMessage("When did it start?"), AIMessage("Is it worse at night?")]))
    graph = build_graph(brain=RunnableLambda(out_of_tokens), backup=backup, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "once"}}

    first = graph.invoke({"messages": [HumanMessage("my calf cramps")]}, config)["messages"][-1]
    second = graph.invoke({"messages": [HumanMessage("last week")]}, config)["messages"][-1]
    assert "backup model" in first.content
    assert second.content == "Is it worse at night?"
    assert second.response_metadata["sympai_backup"] is True  # still marked, just not repeated
