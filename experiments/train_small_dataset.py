import torch

from src.colbert_model import ColBERTModel
from src.loss import pairwise_ranking_loss


TRAINING_DATA = [
    (
        "what is machine learning?",
        "Machine learning is a field of artificial intelligence that learns patterns from data.",
        "The Eiffel Tower is located in Paris, France.",
    ),
    (
        "what is deep learning?",
        "Deep learning uses multi-layer neural networks to learn complex patterns.",
        "The Pacific Ocean is the largest ocean on Earth.",
    ),
    (
        "what is python?",
        "Python is a popular programming language used for software development and data science.",
        "Basketball is played with two teams and a hoop.",
    ),
    (
        "what is artificial intelligence?",
        "Artificial intelligence focuses on building systems that perform tasks requiring human-like intelligence.",
        "Mount Everest is the tallest mountain above sea level.",
    ),
]


def main():
    torch.manual_seed(42)

    model = ColBERTModel(
        model_name="bert-base-uncased",
        projection_dim=128
    )

    # Freeze BERT. Train only the projection layer.
    for param in model.encoder.bert.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(
        model.encoder.projection.parameters(),
        lr=1e-3
    )

    epochs = 10

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0

        for query, positive_doc, negative_doc in TRAINING_DATA:
            optimizer.zero_grad()

            positive_score = model.score(
                query,
                positive_doc
            )

            negative_score = model.score(
                query,
                negative_doc
            )

            loss = pairwise_ranking_loss(
                positive_score.unsqueeze(0),
                negative_score.unsqueeze(0)
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(TRAINING_DATA)

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Average loss: {average_loss:.4f}"
        )

    print("\nEvaluation:\n")

    model.eval()

    with torch.no_grad():
        for query, positive_doc, negative_doc in TRAINING_DATA:
            positive_score = model.score(
                query,
                positive_doc
            )

            negative_score = model.score(
                query,
                negative_doc
            )

            margin = positive_score - negative_score

            print(f"Query: {query}")
            print(
                f"Positive: {positive_score.item():.4f} | "
                f"Negative: {negative_score.item():.4f} | "
                f"Margin: {margin.item():.4f}"
            )
            print()


if __name__ == "__main__":
    main()