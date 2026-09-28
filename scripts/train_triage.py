"""
Step 4b. Train the triage classifier.

Run it once, and again whenever you add examples to triage/examples.csv
    uv run python scripts/train_triage.py

The first run downloads the base model from Hugging Face, about 470 MB, into
~/.cache/huggingface. After that it trains offline in a minute or two on the
M2, then saves the result to models/triage-setfit, which git ignores.

It trains on triage/examples.csv and then tests on triage/test_examples.csv,
sentences it has never seen. Testing on the training data would only show
that it can memorize. The held-out set shows whether it learned.

Training and testing are separate jobs. Training builds a new model from the
examples and saves it. Testing only measures a model. To test the saved model
again without retraining, for example after changing the rules
    uv run python scripts/train_triage.py --evaluate-only
"""

import argparse
import csv
import os
import time
from collections import Counter

# No usage reporting, and quieter logs.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# If an operation is missing on the Mac GPU (MPS), run that one on the CPU instead of failing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from datasets import Dataset  # noqa: E402
from setfit import SetFitModel, Trainer, TrainingArguments  # noqa: E402

from sympai.classifier import BASE_MODEL, EXAMPLES_DIR, LABELS, MODEL_DIR, THRESHOLD  # noqa: E402
from sympai.triage import check  # noqa: E402


def read_examples(name: str) -> list[dict]:
    with open(EXAMPLES_DIR / name, newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row["text"].strip()]
    unknown = {row["label"] for row in rows} - set(LABELS)
    if unknown:
        raise SystemExit(f"{name} has labels {unknown}. Use only {LABELS}.")
    return rows


def as_dataset(rows: list[dict]) -> Dataset:
    return Dataset.from_dict(
        {"text": [row["text"] for row in rows], "label": [LABELS.index(row["label"]) for row in rows]}
    )


def top_label(model: SetFitModel, text: str) -> tuple[str, float]:
    """The model's most likely label and its confidence, before any threshold."""
    probs = model.predict_proba(text, as_numpy=True)
    best = int(probs.argmax())
    return LABELS[int(model.model_head.classes_[best])], float(probs[best])


def desk_level(text: str, top: tuple[str, float], threshold: float) -> str:
    """What the whole triage desk decides, rules first and then the classifier."""
    level = check(text, classifier=lambda _: top, threshold=threshold).level
    return "911" if level == "poison" else level


def evaluate(model: SetFitModel, rows: list[dict]) -> None:
    tops = [top_label(model, row["text"]) for row in rows]

    print(f"The classifier on its own, at threshold {THRESHOLD}")
    print(f"{'label':<7} {'caught':>7} {'recall':>7}")
    for label in LABELS:
        mine = [(row, top) for row, top in zip(rows, tops) if row["label"] == label]
        caught = sum(t[0] == label and (label == "clear" or t[1] >= THRESHOLD) for _, t in mine)
        print(f"{label:<7} {caught:>3}/{len(mine):<3} {caught / max(len(mine), 1):>7.0%}")

    # What matters is what the person experiences, which is the whole desk.
    # Evaluate the system, not just one part of it.
    print("\nThe whole triage desk, rules first and then the classifier")
    print(f"{'threshold':<10} {'missed alerts':>14} {'false alarms':>13}")
    for threshold in (0.5, 0.6, 0.7, 0.8):
        levels = [desk_level(row["text"], top, threshold) for row, top in zip(rows, tops)]
        missed = sum(row["label"] != "clear" and got == "clear" for row, got in zip(rows, levels))
        false = sum(row["label"] == "clear" and got != "clear" for row, got in zip(rows, levels))
        mark = "   <- current" if threshold == THRESHOLD else ""
        print(f"{threshold:<10} {missed:>14} {false:>13}{mark}")

    print(f"\nWhat the whole desk still gets wrong at {THRESHOLD}")
    wrong = 0
    for row, top in zip(rows, tops):
        got = desk_level(row["text"], top, THRESHOLD)
        if got != row["label"]:
            wrong += 1
            print(f"   expected {row['label']:<6} got {got:<6} (classifier said {top[0]} at {top[1]:.2f})  {row['text']}")
    if not wrong:
        print("   nothing")
    print("\nA missed alert is the costly mistake. A false alarm only warns and asks, so lower thresholds are affordable.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and test the triage classifier.")
    parser.add_argument("--evaluate-only", action="store_true", help="test the saved model without retraining")
    options = parser.parse_args()

    test_rows = read_examples("test_examples.csv")
    if options.evaluate_only:
        if not MODEL_DIR.exists():
            raise SystemExit("No saved model yet. Run this script without --evaluate-only first.")
        print(f"Testing the saved model on {len(test_rows)} unseen examples {dict(Counter(r['label'] for r in test_rows))}\n")
        evaluate(SetFitModel.from_pretrained(str(MODEL_DIR)), test_rows)
        return

    train_rows = read_examples("examples.csv")
    print(f"Training on {len(train_rows)} examples {dict(Counter(r['label'] for r in train_rows))}")
    print(f"Testing on {len(test_rows)} unseen examples {dict(Counter(r['label'] for r in test_rows))}\n")

    model = SetFitModel.from_pretrained(BASE_MODEL, labels=LABELS)
    args = TrainingArguments(
        batch_size=16,
        num_epochs=1,
        num_iterations=20,  # pairs made per example. More is slower and not always better.
        report_to="none",  # send nothing to any tracking service
        seed=42,  # same examples in, same model out
    )
    started = time.time()
    Trainer(model=model, args=args, train_dataset=as_dataset(train_rows)).train()
    print(f"\nTrained in {time.time() - started:.0f} seconds\n")

    evaluate(model, test_rows)

    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MODEL_DIR))
    print(f"\nSaved to {MODEL_DIR.relative_to(MODEL_DIR.parents[1])}. The triage desk will use it from the next run.")


if __name__ == "__main__":
    main()
