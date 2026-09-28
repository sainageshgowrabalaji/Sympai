"""
Step 4b. The triage classifier, layer 2 of the triage desk.

The rules in triage.py only know phrasings we wrote down. "It feels like an
elephant is sitting on my chest" means chest pain, but no rule says so. A
classifier learns what emergencies sound like from examples instead.

SetFit trains one from very few examples, about a hundred here, in a couple
of minutes on a laptop. It works in two stages.
1. Contrastive fine-tuning. It makes pairs of examples, "these two mean the
   same kind of thing" and "these two do not", and nudges a small sentence
   embedding model until sentences of the same label land close together.
2. A plain logistic regression on top of those embeddings picks the label.

Classical ML, not an LLM. It runs on your Mac in milliseconds, costs no tokens,
and gives the same answer every time.

It only runs if you have trained it (scripts/train_triage.py) and the ml
dependency group is installed. Otherwise SympAI falls back to the rules alone.
"""

from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "triage-setfit"
EXAMPLES_DIR = ROOT / "triage"

# A small multilingual embedding model, about 470 MB, covering around 100
# languages including Hindi and Telugu. That matters because the examples can
# be in any language the people using SympAI speak.
BASE_MODEL = "intfloat/multilingual-e5-small"
# "urgent" was added after the first real test. With only emergency or clear
# to choose from, the model learned that anything mentioning the chest is 911.
# Real triage has levels in between, and so does this now.
LABELS = ["clear", "urgent", "911", "988"]

# The classifier must be this sure before it raises an alarm on its own. The
# rules still fire at any confidence. Tune this with the test set, not by feel.
THRESHOLD = 0.8


@cache
def _load_model():
    if not MODEL_DIR.exists():
        return None
    import os

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    try:
        from setfit import SetFitModel  # imported here so SympAI runs without the ml group
    except ImportError:
        return None
    model = SetFitModel.from_pretrained(str(MODEL_DIR))
    if list(model.labels or []) != LABELS:
        # Trained before a label was added. Retrain with scripts/train_triage.py.
        return None
    return model


def available() -> bool:
    return _load_model() is not None


def classify(text: str) -> tuple[str, float] | None:
    """The most likely label and how sure it is, or None when there is no trained model."""
    model = _load_model()
    if model is None:
        return None
    probs = model.predict_proba(text, as_numpy=True)
    best = int(probs.argmax())
    labels = model.labels or LABELS
    # The head's classes are the label ids it was trained on, in column order.
    label_id = int(model.model_head.classes_[best])
    return labels[label_id], float(probs[best])
