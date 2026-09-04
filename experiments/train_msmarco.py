import csv

import torch
from torch.utils.data import DataLoader

from src.colbert_model import ColBERTModel
from src.data import ColBERTTripletDataset
from src.loss import pairwise_ranking_loss


DATA_PATH = "data/msmarco_train_100.jsonl"
CSV_PATH = "experiments/msmarco_training_loss.csv"
CHECKPOINT_PATH = "experiments/colbert_msmarco_100.pt"


def collate_fn(batch):
    return {
        "queries": [item["query"] for item in batch],
        "positives": [item["positive"] for item in batch],
        "negatives": [item["negative"] for item in batch],
    }


def main():
    torch.manual_seed(42)

    dataset = ColBERTTripletDataset.from_jsonl(
        DATA_PATH
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = ColBERTModel(
        model_name="bert-base-uncased",
        projection_dim=128
    )

    # CPU-friendly: freeze BERT
    for param in model.encoder.bert.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(
        model.encoder.projection.parameters(),
        lr=1e-3
    )

    epochs = 3

    history = []

    print(
        f"\nTraining on {len(dataset)} MS MARCO triplets...\n"
    )

    for epoch in range(epochs):
        model.train()

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

        average_loss = total_loss / batches

        history.append({
            "epoch": epoch + 1,
            "loss": average_loss
        })

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Average loss: {average_loss:.4f}"
        )

    # Save training loss CSV
    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "loss"]
        )

        writer.writeheader()
        writer.writerows(history)

    # Save checkpoint
    torch.save(
        model.state_dict(),
        CHECKPOINT_PATH
    )

    print(
        f"\nSaved training history to {CSV_PATH}"
    )

    print(
        f"Saved model to {CHECKPOINT_PATH}"
    )


if __name__ == "__main__":
    main()