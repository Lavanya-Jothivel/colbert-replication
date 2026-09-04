import torch

from src.colbert_model import ColBERTModel
from src.indexer import ColBERTIndex


DOCUMENTS = [
    "Machine learning is a field of artificial intelligence that learns patterns from data.",
    "Deep learning uses multi-layer neural networks to learn complex patterns.",
    "Python is a popular programming language used for software development and data science.",
    "Artificial intelligence focuses on building systems that perform tasks requiring human-like intelligence.",
    "The Eiffel Tower is located in Paris, France.",
    "Basketball is a sport played by two teams.",
    "Neural networks are computational models inspired by biological neurons.",
    "Data science combines programming, statistics, and machine learning."
]


QUERIES = [
    "which programming language is used for data science?",
    "how do computers learn patterns from data?",
    "what are neural networks?",
    "where is the Eiffel Tower?"
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

    print("\nLoaded trained ColBERT checkpoint.")

    # Build document index only once
    index = ColBERTIndex()

    index.build(
        model,
        DOCUMENTS
    )

    print(
        f"Indexed {len(index)} documents."
    )

    for query in QUERIES:
        results = index.search(
            model,
            query,
            top_k=3
        )

        print("\n" + "=" * 70)
        print(f"Query: {query}")
        print("=" * 70)

        for rank, result in enumerate(
            results,
            start=1
        ):
            print(
                f"{rank}. Score: "
                f"{result['score']:.4f}"
            )

            print(
                f"   {result['document']}"
            )

            print()


if __name__ == "__main__":
    main()