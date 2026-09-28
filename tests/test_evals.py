"""The eval runner itself, tested offline with fake models for SympAI, the patient and the judge."""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from sympai import evals
from sympai.evals import Verdict, code_checks, run_scenario, triage_kind
from sympai.graph import build_graph


def scenario(**changes):
    base = {
        "id": "x", "who": "test", "first_message": "my calf hurts when I walk", "patient_facts": "You are 30.",
        "expect_triage": "none", "restart_after": "", "correct_if": "asks questions", "wrong_if": "none",
    }
    return {**base, **changes}


def fake_graph_maker(*replies):
    replies = list(replies)

    def make(checkpointer):
        # A fresh fake model per graph, continuing through the same list of replies.
        return build_graph(brain=RunnableLambda(lambda messages: AIMessage(replies.pop(0))), checkpointer=checkpointer)

    return make


def test_a_conversation_plays_until_the_answer():
    patient = GenericFakeChatModel(messages=iter([AIMessage("about two weeks"), AIMessage("no swelling")]))
    make = fake_graph_maker("When did it start?", "Any swelling?", "**Most likely** a strained muscle.")
    outcome = run_scenario(scenario(), make, patient)
    assert [t.person for t in outcome.turns] == ["my calf hurts when I walk", "about two weeks", "no swelling"]
    assert outcome.error is None and all(ok for _, ok, _ in outcome.checks)


def test_an_emergency_stops_after_one_turn():
    outcome = run_scenario(scenario(first_message="my face is drooping", expect_triage="stop-911"), fake_graph_maker(), None)
    assert len(outcome.turns) == 1
    assert triage_kind(outcome.turns[0].triage) == "stop-911"
    assert all(ok for _, ok, _ in outcome.checks)


def test_a_restart_is_remembered():
    patient = GenericFakeChatModel(messages=iter([AIMessage("two weeks"), AIMessage("no")]))
    make = fake_graph_maker("When did it start?", "Any swelling?", "**Most likely** a strain.")
    outcome = run_scenario(scenario(restart_after="1"), make, patient)
    assert ("remembers after a restart", True, "ok") in outcome.checks


def test_code_checks_catch_the_failures_seen_in_testing():
    turn = evals.Turn(
        person="where should I go",
        triage={"level": "clear"},
        sympai=[
            "How old are you? And do you have any conditions?",
            "Go to Westfield Urgent Care at 1000 Central Ave, it costs about $30. Your diagnosis is confirmed.",
        ],
        actions=[],
    )
    failed = {name for name, ok, _ in code_checks(scenario(), [turn]) if not ok}
    assert failed == {"one question at a time", "no invented places, numbers or prices", "never says a diagnosis is confirmed"}


def test_poison_control_is_an_allowed_number():
    turn = evals.Turn("x", {"level": "clear"}, ["Call Poison Control at 1-800-222-1222."], [])
    assert all(ok for _, ok, _ in code_checks(scenario(), [turn]))


def test_the_report_lists_every_case(tmp_path):
    outcome = run_scenario(scenario(first_message="my face is drooping", expect_triage="stop-911"), fake_graph_maker(), None)
    outcome.verdict = Verdict(passed=True, reason="Showed the 911 notice at once.")
    text = evals.report([outcome])
    assert "1 of 1 passed" in text and "Showed the 911 notice" in text



def test_a_place_from_a_tool_is_not_counted_as_invented():
    turn = evals.Turn(
        person="find me an ENT",
        triage={"level": "clear"},
        sympai=["Jane Doe, MD at 10 Main St, Newark. Phone 201-555-0100."],
        actions=[],
        tool_text=["Jane Doe, MD. Otolaryngology\n10 Main St, Newark, NJ 07102\nPhone 201-555-0100"],
    )
    assert all(ok for _, ok, _ in code_checks(scenario(), [turn]))


def test_each_lookup_is_named_after_the_service_it_used():
    from sympai.evals import describe_call

    assert describe_call({"name": "search_pubmed", "args": {"query": "visual snow"}}) == "searched PubMed reviews for 'visual snow'"
    assert describe_call({"name": "search_clinical_trials", "args": {"condition": "visual snow", "location": "New Jersey"}}) == (
        "searched ClinicalTrials.gov for 'visual snow' near New Jersey"
    )
    assert describe_call({"name": "find_clinicians", "args": {"specialty": "ENT", "zip_code": "07102"}}) == (
        "looked up ENT clinicians near 07102 in the NPI Registry"
    )
    assert describe_call({"name": "consult_department", "args": {"department": "ent"}}) == "asked the ENT department"
