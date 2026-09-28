"""
Step 5. See or delete what SympAI remembers about a person.

    uv run python scripts/chart.py show              your chart
    uv run python scripts/chart.py show --person mom
    uv run python scripts/chart.py forget            delete your chart for good
    uv run python scripts/chart.py people            whose charts exist

Everything is in data/charts.sqlite on this laptop. Nothing here talks to Groq.
"""

import argparse

from sympai import chart as charts


def main() -> None:
    parser = argparse.ArgumentParser(description="See or delete a SympAI chart.")
    parser.add_argument("action", choices=["show", "forget", "people"])
    parser.add_argument("--person", default=charts.DEFAULT_PERSON)
    options = parser.parse_args()
    store = charts.open_chart_store()

    if options.action == "people":
        people = sorted({item.namespace[1] for item in store.search(("charts",), limit=1000)})
        print("\n".join(people) or "No charts yet.")
    elif options.action == "show":
        chart = charts.load(store, options.person)
        print(charts.summary(chart) or f"SympAI does not remember anything about {options.person} yet.")
    else:
        charts.forget(store, options.person)
        print(f"Deleted the chart for {options.person}.")


if __name__ == "__main__":
    main()
