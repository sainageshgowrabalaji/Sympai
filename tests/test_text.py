"""Tests for the output guardrails in text.py. Every example is a real question from testing."""

import pytest

from sympai.text import (
    CALL_911_LINE,
    SAME_DAY_LINE,
    clean_answer,
    enforce_minimum_care,
    misrouted_signs,
    one_question,
    route_serious_signs_to_911,
)


@pytest.mark.parametrize(
    "asked, kept",
    [
        ("How old are you, and do you have any medical conditions such as diabetes?", "How old are you?"),
        ("When did the leg pain start and how would you describe its severity?", "When did the leg pain start?"),
        (
            "How long does each episode last, and have you had any recent injuries?",
            "How long does each episode last?",
        ),
        (
            "Where exactly is the pain located (e.g., thigh, calf) and have you noticed any swelling?",
            "Where exactly is the pain located (e.g., thigh, calf)?",
        ),
        ("When did it start? How bad is it?", "When did it start?"),
        (
            "Do you exercise or do you often feel dehydrated or have low potassium or magnesium levels?",
            "Do you exercise?",
        ),
    ],
)
def test_two_questions_become_one(asked, kept):
    assert one_question(asked) == kept


@pytest.mark.parametrize(
    "fine",
    [
        "Have you noticed any swelling, redness, or trouble walking?",  # one yes or no question
        "What is your age range (e.g., 20-29, 30-39, 40-49)?",
        "Do you and your family live in the same house?",
        "How would you describe the pain, sharp, cramping, or achy?",  # options, still one question
        "Is it worse in the morning or at night?",
        "**Most likely**\nA muscle cramp.\n\nDo you want tips on stretching and what to ask a doctor?",
        "You're welcome! Take care.",
    ],
)
def test_single_questions_and_answers_are_untouched(fine):
    assert one_question(fine) == fine


def test_training_citation_marks_are_removed():
    assert clean_answer("See a provider【1†L1-L4】.") == "See a provider."



def test_serious_signs_sent_to_urgent_care_become_911():
    # The exact line from case 3 of the scenario run.
    line = ("- **Urgent care**: Go if the heaviness becomes crushing, lasts >30 minutes, "
            "or is accompanied by new shortness of breath, sweating, nausea, or dizziness.")
    fixed, changed = route_serious_signs_to_911("**Where to go**\n" + line)
    assert changed
    assert fixed.splitlines()[1].startswith("- **Call 911, not urgent care**, if you have crushing chest pain or pressure")
    assert misrouted_signs(fixed.splitlines()[1]) == []


def test_ordinary_urgent_care_advice_is_left_alone():
    for line in ["- **Urgent care**: go if the pain gets worse or you get a fever.",
                 "Call 911, not urgent care, if you have shortness of breath."]:
        assert route_serious_signs_to_911(line) == (line, False)


def test_the_triage_desks_own_stroke_notice_is_not_flagged():
    notice = ("Sudden weakness or numbness, a drooping face, trouble speaking or sudden vision loss can be signs "
              "of a stroke. Stroke treatment works best in the first hours, and urgent care cannot give it.")
    assert misrouted_signs(notice) == []



def test_a_hyphenated_urgent_care_is_caught_too():
    # The line from the third scenario run.
    line = ("If the pressure worsens or you develop shortness of breath, schedule a same-day or "
            "urgent-care visit with your primary care provider.")
    assert misrouted_signs(line) == ["shortness of breath"]


def test_an_urgent_conversation_gets_same_day_care_and_a_911_line():
    answer = "**Most likely**\nA muscle strain.\n\n**Where to go**\nStretch, and see a doctor if it lasts 2 weeks."
    fixed, changed = enforce_minimum_care(answer, "urgent")
    assert changed and SAME_DAY_LINE in fixed and CALL_911_LINE in fixed


def test_minimum_care_leaves_other_replies_alone():
    assert enforce_minimum_care("When did it start?", "urgent") == ("When did it start?", False)
    assert enforce_minimum_care("**Most likely** a cold.", "clear") == ("**Most likely** a cold.", False)
    already = "**Most likely** a strain.\nSee primary care today. Call 911 if it gets severe."
    assert enforce_minimum_care(already, "911") == (already, False)



def test_unicode_hyphens_do_not_hide_anything():
    # "Same\u2011day" and "urgent\u2011care" as the model really wrote them.
    already = "**Most likely** a strain.\n- **Same\u2011day** primary care visit. Call 911 if it gets severe."
    assert enforce_minimum_care(already, "urgent") == (already, False)
    assert misrouted_signs("Go to urgent\u2011care if you get shortness of breath.") == ["shortness of breath"]



def test_a_kind_opening_line_does_not_hide_two_questions():
    # The first reply in the chat screen test.
    reply = ("I'm sorry you've been dealing with uncomfortable vertigo and tinnitus, it sounds frustrating.\n\n"
             "When did the most recent vertigo episodes start, and how long did each episode last?")
    assert one_question(reply) == (
        "I'm sorry you've been dealing with uncomfortable vertigo and tinnitus, it sounds frustrating.\n\n"
        "When did the most recent vertigo episodes start?"
    )
