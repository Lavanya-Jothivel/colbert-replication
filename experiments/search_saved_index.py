import torch

from src.colbert_model import ColBERTModel
from src.indexer import ColBERTIndex


def main():
    model = ColBERTModel(
        model_name="bert-base-uncased",
        projection_dim=128
    )

    state_dict = torch.load(
        "experiments/colbert_tiny.pt",
        map_location="cpu"
    )

    model.load_state_dict(state_dict)
    model.eval()

    index = ColBERTIndex()

    index.load(
        "experiments/colbert_index.pt"
    )

    print(f"\nLoaded index with {len(index)} documents.")

    query = "which programming language is popular for data science?"

    results = index.search(
        model,
        query,
        top_k=3
    )

    print(f"\nQuery: {query}\n")

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. Score: {result['score']:.4f}"
        )
        print(
            f"   {result['document']}"
        )
        print()


if __name__ == "__main__":
    main()