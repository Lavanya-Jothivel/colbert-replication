import torch

from src.colbert_model import ColBERTModel


DOCUMENTS = [
    "Machine learning is a field of artificial intelligence that learns patterns from data.",
    "Deep learning uses multi-layer neural networks to learn complex patterns.",
    "Python is a popular programming language used for software development and data science.",
    "Artificial intelligence focuses on building systems that perform tasks requiring human-like intelligence.",
    "The Eiffel Tower is located in Paris, France.",
    "Basketball is a sport played by two teams."
]


def main():
    model = ColBERTModel(
        model_name="bert-base-uncased",
        projection_dim=128
    )

    checkpoint_path = "experiments/colbert_tiny.pt"

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    model.load_state_dict(state_dict)

    model.eval()

    query = "which programming language is used for data science?"

    with torch.no_grad():
        scores = model.score_matrix(
            [query],
            DOCUMENTS
        )[0]

    ranked_indices = torch.argsort(
        scores,
        descending=True
    )

    print("\nQuery:")
    print(query)

    print("\nRanking:\n")

    for rank, index in enumerate(ranked_indices, start=1):
        index = index.item()

        print(
            f"{rank}. Score: {scores[index].item():.4f}"
        )
        print(
            f"   {DOCUMENTS[index]}"
        )
        print()


if __name__ == "__main__":
    main()