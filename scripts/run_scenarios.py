"""
Step 10. Play every test case in evals/scenarios.csv against SympAI, by itself.

Run it
    uv run python scripts/run_scenarios.py            all 11 cases, about 10 to 15 minutes
    uv run python scripts/run_scenarios.py --only 3,7  just those cases

It prints PASS or FAIL per case as it goes, and writes every transcript with
its checks and the judge's reason to evals/results/latest.md.

Cost. The whole run uses roughly a third to a half of the day's free
gpt-oss-120b allowance on Groq. Most of the wait is Groq's 8,000 tokens a
minute limit, which the client handles by waiting and retrying.
"""

import argparse

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.store.memory import InMemoryStore

from sympai.departments import MAX_RETRIES
from sympai.evals import JUDGE_MODEL, PATIENT_MODEL, Verdict, run_all
from sympai.graph import build_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Play the test cases against SympAI.")
    parser.add_argument("--only", help="case ids to run, like 3,7")
    options = parser.parse_args()
    only = {part.strip() for part in options.only.split(",")} if options.only else None

    load_dotenv()
    patient = ChatGroq(model=PATIENT_MODEL, temperature=0.3, reasoning_effort="low", max_retries=MAX_RETRIES)
    judge_model = ChatGroq(
        model=JUDGE_MODEL, temperature=0, reasoning_effort="medium", max_retries=MAX_RETRIES
    ).with_structured_output(Verdict, method="json_schema", strict=True)
    # In the first run the judge's JSON failed once on Groq's side, and that
    # case was marked FAIL although SympAI had done nothing wrong. A second
    # judge on a different model now takes over when that happens.
    backup_judge = ChatGroq(
        model="qwen/qwen3.8-27b", temperature=0, reasoning_format="parsed", max_retries=MAX_RETRIES
    ).with_structured_output(Verdict, method="json_schema")
    judge_model = judge_model.with_fallbacks([backup_judge])

    # The real SympAI graph, the same one chat.py runs, with a fresh saved file per case.
    # Charts go to a throwaway in-memory store, never into your real data/charts.sqlite.
    test_charts = InMemoryStore()
    run_all(
        only,
        lambda checkpointer: build_graph(checkpointer=checkpointer, chart_store=test_charts),
        patient,
        judge_model,
    )


if __name__ == "__main__":
    main()
