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

    index.build(
        model,
        DOCUMENTS
    )

    index_path = "experiments/colbert_index.pt"

    index.save(index_path)

    print(f"\nIndexed {len(index)} documents.")
    print(f"Saved index to {index_path}")


if __name__ == "__main__":
    main()