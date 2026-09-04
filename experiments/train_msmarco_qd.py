import csv
import json

import torch
from torch.utils.data import DataLoader

from src.colbert_model import ColBERTModel
from src.data import ColBERTTripletDataset
from src.loss import pairwise_ranking_loss


DATA_PATH = "data/msmarco_train_1000.jsonl"
CHECKPOINT_PATH = "experiments/colbert_msmarco_qd_800.pt"
CSV_PATH = "results/msmarco_qd_800_200_metrics.csv"


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

    train_examples = examples[:800]
    validation_examples = examples[800:]

    train_dataset = ColBERTTripletDataset(
        train_examples
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = ColBERTModel(
        model_name="bert-base-uncased",
        projection_dim=128
    )

    # Freeze BERT
    for param in model.encoder.bert.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(
        model.encoder.projection.parameters(),
        lr=5e-4
    )

    epochs = 3
    history = []

    print("\nColBERT [Q]/[D] training")
    print("Training examples: 800")
    print("Validation examples: 200\n")

    for epoch in range(epochs):
        model.train()

        # Frozen BERT should stay deterministic
        model.encoder.bert.eval()

        total_loss = 0.0
        batches = 0

        for batch_index, batch in enumerate(
            dataloader,
            start=1
        ):
            optimizer.zero_grad()

            positive_scores = []
            negative_scores = []

            for query, positive, negative in zip(
                batch["queries"],
                batch["positives"],
                batch["negatives"]
            ):
                positive_score = model.score(
                    query,
                    positive
                )

                negative_score = model.score(
                    query,
                    negative
                )

                positive_scores.append(
                    positive_score
                )

                negative_scores.append(
                    negative_score
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

            if batch_index % 20 == 0:
                print(
                    f"Epoch {epoch + 1} | "
                    f"Batch {batch_index}/{len(dataloader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        average_loss = total_loss / batches

        validation_accuracy, validation_margin = evaluate(
            model,
            validation_examples
        )

        history.append({
            "epoch": epoch + 1,
            "training_loss": average_loss,
            "validation_accuracy": validation_accuracy,
            "validation_margin": validation_margin
        })

        print("\n-----------------------------")
        print(
            f"Epoch {epoch + 1} complete"
        )
        print(
            f"Training loss: "
            f"{average_loss:.4f}"
        )
        print(
            f"Validation accuracy: "
            f"{validation_accuracy * 100:.2f}%"
        )
        print(
            f"Validation margin: "
            f"{validation_margin:.4f}"
        )
        print("-----------------------------\n")

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

    print(
        f"Saved model to {CHECKPOINT_PATH}"
    )

    print(
        f"Saved metrics to {CSV_PATH}"
    )


if __name__ == "__main__":
    main()