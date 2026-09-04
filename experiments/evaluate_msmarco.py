import json

import torch

from src.colbert_model import ColBERTModel


DATA_PATH = "data/msmarco_train_100.jsonl"

CHECKPOINT_PATH = (
    "experiments/colbert_msmarco_100.pt"
)


def load_examples():
    examples = []

    with open(
        DATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:
            examples.append(
                json.loads(line)
            )

    return examples


def main():
    examples = load_examples()

    # Use the last 20 examples as a simple
    # held-out evaluation set
    evaluation_examples = examples[-20:]

    model = ColBERTModel(
        model_name="bert-base-uncased",
        projection_dim=128
    )

    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu"
    )

    model.load_state_dict(state_dict)
    model.eval()

    correct = 0

    margins = []

    print(
        "\nEvaluating MS MARCO checkpoint...\n"
    )

    with torch.no_grad():

        for example in evaluation_examples:

            positive_score = model.score(
                example["query"],
                example["positive"]
            )

            negative_score = model.score(
                example["query"],
                example["negative"]
            )

            margin = (
                positive_score - negative_score
            ).item()

            margins.append(margin)

            if positive_score > negative_score:
                correct += 1

    total = len(evaluation_examples)

    accuracy = correct / total

    average_margin = (
        sum(margins) / len(margins)
    )

    print(
        f"Pairwise accuracy: "
        f"{correct}/{total} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Average score margin: "
        f"{average_margin:.4f}"
    )


if __name__ == "__main__":
    main()