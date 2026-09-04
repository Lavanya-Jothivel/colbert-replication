import torch

from src.colbert_model import ColBERTModel
from src.batch_loss import inbatch_cross_entropy_loss


QUERIES = [
    "what is machine learning?",
    "what is deep learning?",
    "what is python?",
    "what is artificial intelligence?"
]

DOCUMENTS = [
    "Machine learning is a field of artificial intelligence that learns patterns from data.",
    "Deep learning uses multi-layer neural networks to learn complex patterns.",
    "Python is a popular programming language used for software development and data science.",
    "Artificial intelligence focuses on building systems that perform tasks requiring human-like intelligence."
]


def main():
    torch.manual_seed(42)

    model = ColBERTModel(
        model_name="bert-base-uncased",
        projection_dim=128
    )

    # Freeze BERT for CPU-friendly training
    for param in model.encoder.bert.parameters():
        param.requires_grad = False

    # Train only the projection layer
    optimizer = torch.optim.Adam(
        model.encoder.projection.parameters(),
        lr=1e-3
    )

    epochs = 15

    print("\nStarting in-batch training...\n")

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()

        # Score every query against every document
        scores = model.score_matrix(
            QUERIES,
            DOCUMENTS
        )

        # Correct documents are on the diagonal
        loss = inbatch_cross_entropy_loss(scores)

        loss.backward()

        optimizer.step()

        predictions = scores.argmax(dim=1)

        targets = torch.arange(
            len(QUERIES),
            device=scores.device
        )

        accuracy = (
            predictions == targets
        ).float().mean()

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Loss: {loss.item():.4f} | "
            f"Accuracy: {accuracy.item() * 100:.1f}%"
        )

    # --------------------------------------------------
    # Final evaluation
    # --------------------------------------------------

    print("\nFinal score matrix:\n")

    model.eval()

    with torch.no_grad():
        scores = model.score_matrix(
            QUERIES,
            DOCUMENTS
        )

    print(scores)

    predictions = scores.argmax(dim=1)

    print("\nRanking results:\n")

    correct = 0

    for i, query in enumerate(QUERIES):
        predicted_index = predictions[i].item()
        correct_index = i

        if predicted_index == correct_index:
            correct += 1

        print(f"Query: {query}")
        print(f"Predicted document: {predicted_index + 1}")
        print(f"Correct document:   {correct_index + 1}")
        print()

    accuracy = correct / len(QUERIES)

    print(
        f"Final retrieval accuracy: "
        f"{correct}/{len(QUERIES)} "
        f"({accuracy * 100:.1f}%)"
    )

    # --------------------------------------------------
    # Save trained model
    # --------------------------------------------------

    checkpoint_path = "experiments/colbert_tiny.pt"

    torch.save(
        model.state_dict(),
        checkpoint_path
    )

    print(
        f"\nSaved model to {checkpoint_path}"
    )


if __name__ == "__main__":
    main()