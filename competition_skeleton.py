import os
import random
from pathlib import Path
from typing import Literal

import dspy
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_BASE = "https://ucslab.online/v1"
API_KEY = os.environ["LAB_API_KEY"]

gemma = dspy.LM(
    "openai/google/gemma-4-E2B-it",
    api_base=API_BASE,
    api_key=API_KEY,
    max_tokens=1000,
)
gemma_big = dspy.LM(
    "openai/google/gemma-4-31B-it",
    api_base=API_BASE,
    api_key=API_KEY,
    max_tokens=1000,
)
dspy.configure(lm=gemma)

DATA_DIR = Path("student")
OUTPUT_PATH = "competition_guardrail.json"


def load_examples(csv_name: str = "train.csv") -> list[dspy.Example]:
    df = pd.read_csv(DATA_DIR / csv_name)
    return [
        dspy.Example(
            image=dspy.Image(str(DATA_DIR / row.image)),
            prompt=row.prompt,
            label=row.label,
        ).with_inputs("image", "prompt")
        for row in df.itertuples()
    ]


def split_examples(
    examples: list[dspy.Example], val_frac: float = 0.2, seed: int = 0
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Hold out a validation split so you can check your guardrail before submitting.

    The official test set is not released to students, so this local split is your
    best estimate of how the guardrail will perform on unseen examples.
    """
    shuffled = examples.copy()
    random.Random(seed).shuffle(shuffled)
    n_val = int(len(shuffled) * val_frac)
    return shuffled[n_val:], shuffled[:n_val]


class GuardrailSignature(dspy.Signature):
    """Decide whether a user's image+text message is an attack or benign."""

    image: dspy.Image = dspy.InputField(desc="the image the user attached")
    prompt: str = dspy.InputField(desc="the user's accompanying text")
    label: Literal["attack", "benign"] = dspy.OutputField()


def label_accuracy(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> bool:
    return example.label == prediction.label


def evaluate(guardrail: dspy.Module, devset: list[dspy.Example]) -> float:
    """Run the guardrail over `devset` and return accuracy as a percentage."""
    evaluator = dspy.Evaluate(devset=devset, metric=label_accuracy, display_progress=True, display_table=5)
    result = evaluator(guardrail)
    return result.score


def save_guardrail(guardrail: dspy.Module, path: str = OUTPUT_PATH) -> None:
    guardrail.save(path)


def load_guardrail(path: str = OUTPUT_PATH) -> dspy.Module:
    guardrail = dspy.ChainOfThought(GuardrailSignature)
    guardrail.load(path)
    return guardrail


if __name__ == "__main__":
    trainset, valset = split_examples(load_examples())

    final_guardrail = dspy.ChainOfThought(GuardrailSignature)
    final_guardrail.set_lm(gemma)

    # TODO: optimize final_guardrail on `trainset` (e.g. with a DSPy optimizer) before evaluating.

    accuracy = evaluate(final_guardrail, valset)
    print(f"validation accuracy: {accuracy:.1f}% ({len(trainset)} train / {len(valset)} val examples)")

    save_guardrail(final_guardrail, OUTPUT_PATH)
    print(f"saved {OUTPUT_PATH}")
