import csv
import json

import torch
from torch.utils.data import DataLoader

from src.colbert_model import ColBERTModel
from src.data import ColBERTTripletDataset
from src.loss import pairwise_ranking_loss


DATA_PATH = "data/msmarco_train_100.jsonl"
CHECKPOINT_PATH = "experiments/colbert_msmarco_80.pt"
CSV_PATH = "results/msmarco_split_metrics.csv"


def collate_fn(batch):
    return {
        "queries": [item["query"] for item in batch],
        "positives": [item["positive"] for item in batch],
        "negatives": [item["negative"] for item in batch],
    }


def load_examples():
    examples = []

    with open(DATA_PATH, "r", encoding="utf-8") as file:
        for line in file:
            examples.append(json.loads(line))

    return examples


def evaluate(model, examples):
    model.eval()

    correct = 0
    margins = []

    with torch.no_grad():
        for example in examples:
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

    accuracy = correct / len(examples)
    average_margin = sum(margins) / len(margins)

    return accuracy, average_margin


def main():
    torch.manual_seed(42)

    examples = load_examples()

    train_examples = examples[:80]
    validation_examples = examples[80:]

    train_dataset = ColBERTTripletDataset(
        train_examples
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = ColBERTModel(
        model_name="bert-base-uncased",
        projection_dim=128
    )

    for param in model.encoder.bert.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(
        model.encoder.projection.parameters(),
        lr=1e-3
    )

    epochs = 5
    history = []

    print("\nTraining: 80 examples")
    print("Validation: 20 unseen examples\n")

    for epoch in range(epochs):
        model.train()

# BERT is frozen, so disable its dropout during training.
        model.encoder.bert.eval()

        total_loss = 0.0
        batches = 0

        for batch in dataloader:
            optimizer.zero_grad()

            positive_scores = []
            negative_scores = []

            for query, positive, negative in zip(
                batch["queries"],
                batch["positives"],
                batch["negatives"]
            ):
                positive_scores.append(
                    model.score(query, positive)
                )

                negative_scores.append(
                    model.score(query, negative)
                )

            positive_scores = torch.stack(
                positive_scores
            )

            negative_scores = torch.stack(
                negative_scores
            )

            loss = pairwise_ranking_loss(
                positive_scores,
                negative_scores
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        training_loss = total_loss / batches

        validation_accuracy, validation_margin = evaluate(
            model,
            validation_examples
        )

        history.append({
            "epoch": epoch + 1,
            "training_loss": training_loss,
            "validation_accuracy": validation_accuracy,
            "validation_margin": validation_margin
        })

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Loss: {training_loss:.4f} | "
            f"Val Accuracy: "
            f"{validation_accuracy * 100:.2f}% | "
            f"Val Margin: {validation_margin:.4f}"
        )

    torch.save(
        model.state_dict(),
        CHECKPOINT_PATH
    )

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "training_loss",
                "validation_accuracy",
                "validation_margin"
            ]
        )

        writer.writeheader()
        writer.writerows(history)

    print(f"\nSaved model to {CHECKPOINT_PATH}")
    print(f"Saved metrics to {CSV_PATH}")


if __name__ == "__main__":
    main()