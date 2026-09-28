"""
Departments as data.

There is ONE department agent in code. Each department is a YAML file in
departments/ that it reads. The agent gets the file's instructions, looks at
the case, and returns a DepartmentReport, a typed object (a DTO in Java
terms) rather than free text. Typed output means the coordinator, the tests
and later the chart can all rely on the same fields being there.
"""

from functools import cache
from pathlib import Path
from typing import Literal

import yaml
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, model_validator

DEPARTMENTS_DIR = Path(__file__).resolve().parents[2] / "departments"

# Every department runs on the same model with its own instructions. Using a
# different model from the coordinator also means a separate free-tier
# allowance, so a busy consult does not starve the coordinator.
DEPARTMENT_MODEL = "qwen/qwen3.8-27b"
# Qwen 3.8 is a preview model on Groq, which can be busy or retired. If the
# call fails for any reason, the same request goes to this one instead.
BACKUP_MODEL = "openai/gpt-oss-20b"
# The Groq client waits and retries when it gets a 429 (rate limit). The
# default is 2 tries, which was not enough in testing, so we allow more.
MAX_RETRIES = 6


class DepartmentConfig(BaseModel):
    """The shape every departments/*.yaml file must have. A typo fails the tests."""

    name: str
    scope: str
    red_flags: list[str]
    asks_about: list[str]
    usual_tests: list[str]
    refers_to: dict[str, str]
    sources: list[str]


class Finding(BaseModel):
    name: str = Field(description="A possible condition, in plain words")
    why: str = Field(description="What in the case points to it, in one sentence")
    confidence: Literal["high", "medium", "low"]


class DepartmentReport(BaseModel):
    department: str
    in_scope: bool = Field(description="False if the case is outside this department's scope")
    most_likely: list[Finding]
    also_possible: list[Finding]
    red_flags_seen: list[str] = Field(description="Warning signs present in the case. Empty if none")
    what_would_confirm: list[str] = Field(description="Exams or tests a doctor may use")
    refer_to: list[str] = Field(description="Other departments that should see this, and why")
    sources: list[str] = Field(description="Only URLs that appear in the case notes. Never invent one")

    @model_validator(mode="after")
    def in_scope_means_an_opinion(self):
        """Both real ENT reports in testing came back in scope with no findings.
        That report is useless to the coordinator, so it now counts as invalid,
        the same way Bean Validation rejects a DTO. The parser raises, and
        consult() retries on the backup model."""
        if self.in_scope and not (self.most_likely or self.also_possible):
            raise ValueError("an in-scope report must name at least one finding")
        return self


@cache
def load_departments() -> dict[str, DepartmentConfig]:
    """Read every YAML file once. The key is the file name, like 'ent'."""
    return {
        path.stem: DepartmentConfig(**yaml.safe_load(path.read_text()))
        for path in sorted(DEPARTMENTS_DIR.glob("*.yaml"))
    }


def department_prompt(config: DepartmentConfig) -> str:
    """Turn a config file into the department's system prompt."""
    bullets = lambda items: "\n".join(f"- {item}" for item in items)  # noqa: E731
    referrals = "\n".join(f"- {dept} for {why}" for dept, why in config.refers_to.items())
    return f"""You are the {config.name} department in SympAI's hospital.
You give a specialist opinion to the coordinator, never to the patient.

Your scope is {config.scope}.

Red flags in your field
{bullets(config.red_flags)}

What matters in your field
{bullets(config.asks_about)}

Usual tests
{bullets(config.usual_tests)}

Refer to
{referrals}

Rules
- in_scope is true when ANY part of the case is in your scope. Judge that part, even when the rest belongs to other departments, and refer the rest.
- When in_scope is true, give at least one finding in most_likely. Use low confidence when the information is thin, and say what is missing in why.
- Only when nothing in the case is in your scope, set in_scope to false, leave the findings empty, and refer.
- Never call anything confirmed. Use high confidence only for a classic pattern with nothing pointing elsewhere.
- Put every red flag you see in red_flags_seen.
- Put in sources only URLs that appear in the case notes. Never invent one."""


@cache
def _department_brains():
    """The department model and its backup, both asked for a DepartmentReport.

    json_schema with strict=True makes Groq constrain the output to the
    schema, so every field is there and valid. One catch we found by reading
    what the library actually sends. langchain-groq 1.1.3 only allows strict
    mode for the two gpt-oss models, so for Qwen it quietly drops to
    best-effort. The Pydantic parser still validates Qwen's JSON, and if it
    is wrong the backup retries on gpt-oss-20b, which does get strict mode.
    """
    primary = ChatGroq(
        model=DEPARTMENT_MODEL, temperature=0, reasoning_format="parsed", max_retries=MAX_RETRIES
    ).with_structured_output(DepartmentReport, method="json_schema", strict=True)
    backup = ChatGroq(
        model=BACKUP_MODEL, temperature=0, reasoning_effort="medium", max_retries=MAX_RETRIES
    ).with_structured_output(DepartmentReport, method="json_schema", strict=True)
    return primary, backup


def consult(department: str, case_summary: str) -> tuple[DepartmentReport, str]:
    """Ask one department for its opinion. Returns the report and which model wrote it.

    primary.with_fallbacks([backup]) does the same in one line, but it hides
    which model answered. While we are still debugging we want to see it, so
    the fallback is written out as a plain try and except.
    """
    config = load_departments()[department]
    messages = [("system", department_prompt(config)), ("human", case_summary)]
    primary, backup = _department_brains()
    try:
        return primary.invoke(messages), DEPARTMENT_MODEL
    except Exception as error:
        why = type(error).__name__
    try:
        return backup.invoke(messages), f"{BACKUP_MODEL}, after {DEPARTMENT_MODEL} failed with {why}"
    except Exception as error:
        # In the calf test the backup's JSON left out four required fields, and
        # Groq rejected it. The same question passed on the next turn, because
        # each generation is different, so one more try is worth it here.
        if "json_validate_failed" not in str(error):
            raise
    return backup.invoke(messages), f"{BACKUP_MODEL} on a second try, after {DEPARTMENT_MODEL} failed with {why}"


def departments_for_coordinator() -> str:
    """One line per department, for the coordinator's tool description."""
    return "\n".join(f"- {key}: {cfg.scope}" for key, cfg in load_departments().items())
