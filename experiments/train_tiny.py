import torch

from src.colbert_model import ColBERTModel
from src.loss import pairwise_ranking_loss


def main():
    torch.manual_seed(42)

    model = ColBERTModel(
        model_name="bert-base-uncased",
        projection_dim=128
    )

    # Freeze BERT parameters
    for param in model.encoder.bert.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(
        model.encoder.projection.parameters(),
        lr=1e-3
    )

    query = "what is machine learning?"

    positive_document = (
        "Machine learning is a field of artificial intelligence "
        "that learns patterns from data."
    )

    negative_document = (
        "The Eiffel Tower is located in Paris, France."
    )

    print("\nBefore training:")

    model.eval()

    with torch.no_grad():
        positive_before = model.score(
            query,
            positive_document
        )

        negative_before = model.score(
            query,
            negative_document
        )

    print(f"Positive score: {positive_before.item():.4f}")
    print(f"Negative score: {negative_before.item():.4f}")

    model.train()

    for epoch in range(20):
        optimizer.zero_grad()

        positive_score = model.score(
            query,
            positive_document
        )

        negative_score = model.score(
            query,
            negative_document
        )

        loss = pairwise_ranking_loss(
            positive_score.unsqueeze(0),
            negative_score.unsqueeze(0)
        )

        loss.backward()

        optimizer.step()

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Loss: {loss.item():.4f} | "
            f"Positive: {positive_score.item():.4f} | "
            f"Negative: {negative_score.item():.4f}"
        )

    print("\nAfter training:")

    model.eval()

    with torch.no_grad():
        positive_after = model.score(
            query,
            positive_document
        )

        negative_after = model.score(
            query,
            negative_document
        )

    print(f"Positive score: {positive_after.item():.4f}")
    print(f"Negative score: {negative_after.item():.4f}")

    print(
        f"Margin: "
        f"{(positive_after - negative_after).item():.4f}"
    )


if __name__ == "__main__":
    main()