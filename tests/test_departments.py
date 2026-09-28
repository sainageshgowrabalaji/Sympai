"""Departments are data, so the tests check the data."""

import pytest
from pydantic import ValidationError

from sympai.departments import DepartmentReport, department_prompt, load_departments


def test_every_department_file_is_valid():
    departments = load_departments()
    assert "ent" in departments
    for key, config in departments.items():
        assert config.red_flags, f"{key}.yaml has no red flags"


def test_the_prompt_carries_the_whole_config():
    ent = load_departments()["ent"]
    prompt = department_prompt(ent)
    assert ent.scope in prompt
    assert all(flag in prompt for flag in ent.red_flags)
    assert all(test in prompt for test in ent.usual_tests)


def test_every_report_field_is_required():
    # Groq's strict mode only works when the schema requires every field.
    schema = DepartmentReport.model_json_schema()
    assert set(schema["required"]) == set(schema["properties"])


def _report(**changes):
    fields = dict(
        department="ENT", in_scope=True, most_likely=[], also_possible=[],
        red_flags_seen=[], what_would_confirm=[], refer_to=[], sources=[],
    )
    return DepartmentReport(**{**fields, **changes})


def test_an_in_scope_report_with_no_findings_is_rejected():
    # The exact report ENT sent back twice in testing.
    with pytest.raises(ValidationError):
        _report(in_scope=True)


def test_an_out_of_scope_report_may_be_empty():
    assert _report(in_scope=False, refer_to=["neurology"]).in_scope is False



def test_every_referral_points_to_a_real_department():
    # Like a foreign key. A typo in refers_to would send the coordinator nowhere.
    departments = load_departments()
    for key, config in departments.items():
        for target in config.refers_to:
            assert target in departments, f"{key}.yaml refers to '{target}', which has no file"


def test_the_starter_set_is_all_there():
    assert set(load_departments()) == {
        "general_medicine", "heart_lungs", "neurology", "ent", "eyes", "skin", "bones_muscles", "mental_health",
    }


class _Brain:
    """Answers from a list, one item per call. An exception in the list is raised."""

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), 0

    def invoke(self, messages):
        self.calls += 1
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def _cramp_report():
    from sympai.departments import DepartmentReport, Finding

    return DepartmentReport(
        department="bones_muscles", in_scope=True,
        most_likely=[Finding(name="Muscle cramps", why="Cramping with no swelling", confidence="low")],
        also_possible=[], red_flags_seen=[], what_would_confirm=[], refer_to=[], sources=[],
    )


def test_a_bad_json_answer_from_the_backup_gets_one_more_try(monkeypatch):
    from sympai import departments

    primary = _Brain(RuntimeError("429 rate limit"))
    backup = _Brain(RuntimeError("Error code 400 json_validate_failed, missing properties"), _cramp_report())
    monkeypatch.setattr(departments, "_department_brains", lambda: (primary, backup))
    report, answered_by = departments.consult("bones_muscles", "calf cramps")
    assert report.most_likely[0].name == "Muscle cramps"
    assert backup.calls == 2 and "second try" in answered_by


def test_other_backup_failures_are_not_retried(monkeypatch):
    import pytest
    from sympai import departments

    backup = _Brain(RuntimeError("429 rate limit"), _cramp_report())
    monkeypatch.setattr(departments, "_department_brains", lambda: (_Brain(RuntimeError("429")), backup))
    with pytest.raises(RuntimeError):
        departments.consult("bones_muscles", "calf cramps")
    assert backup.calls == 1
