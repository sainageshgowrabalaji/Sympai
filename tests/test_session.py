"""
Tests for pausing and resuming. Each test uses a throwaway SQLite file, so a
"restart" is simply a second graph opened on the same file.
"""

from datetime import datetime, timedelta, timezone

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from sympai.graph import build_graph
from sympai.session import last_thread, open_checkpointer, recap, remember_thread, thread_config, time_ago


def test_a_conversation_survives_a_restart(tmp_path):
    db = tmp_path / "conversations.sqlite"
    config = thread_config("friend-1")

    first_run = build_graph(
        brain=GenericFakeChatModel(messages=iter([AIMessage("When did the ringing get louder?")])),
        checkpointer=open_checkpointer(db),
    )
    first_run.invoke({"messages": [HumanMessage("My tinnitus is louder")]}, config)

    # The script ends, the laptop sleeps, and later a brand new process starts.
    second_run = build_graph(
        brain=GenericFakeChatModel(messages=iter([AIMessage("Thanks. Is it in one ear or both?")])),
        checkpointer=open_checkpointer(db),
    )
    where = recap(second_run.get_state(config))
    assert where["topic"] == "My tinnitus is louder"
    assert where["last_said"] == "When did the ringing get louder?"
    assert where["unfinished"] is False

    after = second_run.invoke({"messages": [HumanMessage("About six weeks ago")]}, config)
    assert [m.type for m in after["messages"]] == ["human", "ai", "human", "ai"]


def test_a_turn_that_failed_can_be_finished_later(tmp_path):
    calls = {"n": 0}

    def flaky_model(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            # A dropped connection. Rate limits no longer crash, they get the backup model or a "come back" note.
            raise ConnectionError("Groq did not answer")
        return AIMessage("Sorry for the wait. How old are you?")

    graph = build_graph(brain=RunnableLambda(flaky_model), checkpointer=open_checkpointer(tmp_path / "c.sqlite"))
    config = thread_config("friend-2")

    try:
        graph.invoke({"messages": [HumanMessage("Visual snow")]}, config)
    except ConnectionError:
        pass
    # The person's message was saved, and the coordinator is still due to run.
    assert recap(graph.get_state(config))["unfinished"] is True

    done = graph.invoke(None, config)  # None means carry on from the checkpoint
    assert done["messages"][-1].content == "Sorry for the wait. How old are you?"
    assert [m.type for m in done["messages"]] == ["human", "ai"]


def test_the_last_thread_is_remembered_per_person(tmp_path):
    path = tmp_path / "last_thread.json"
    assert last_thread("me", path) is None
    remember_thread("abc", "me", path)
    remember_thread("xyz", "mom", path)
    assert last_thread("me", path) == "abc"
    assert last_thread("mom", path) == "xyz"


def test_the_old_single_thread_file_still_works(tmp_path):
    path = tmp_path / "last_thread.json"
    path.write_text('{"thread_id": "old"}')
    assert last_thread("me", path) == "old"


def test_time_ago_reads_naturally():
    now = datetime(2026, 9, 26, 20, 0, tzinfo=timezone.utc)
    assert time_ago((now - timedelta(minutes=30)).isoformat(), now) == "30 minutes ago"
    assert time_ago((now - timedelta(hours=5)).isoformat(), now) == "5 hours ago"
    assert time_ago((now - timedelta(days=3)).isoformat(), now) == "3 days ago"
