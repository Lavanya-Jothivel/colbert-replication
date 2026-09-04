import torch

from src.colbert_model import ColBERTModel
from src.indexer import ColBERTIndex
from src.metrics import (
    mean_reciprocal_rank,
    mean_recall_at_k
)


QUERIES = [
    "which programming language is used for data science?",
    "how do computers learn patterns from data?",
    "what are neural networks?",
    "where is the Eiffel Tower?"
]

# Correct document indices in the saved 8-document index
RELEVANT_INDICES = [
    2,
    0,
    6,
    4
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

    index.load(
        "experiments/colbert_index.pt"
    )

    all_rankings = []

    print("\nRetrieval Evaluation\n")

    for query in QUERIES:
        results = index.search(
            model,
            query,
            top_k=len(index)
        )

        ranked_indices = [
            result["index"]
            for result in results
        ]

        all_rankings.append(
            ranked_indices
        )

    mrr = mean_reciprocal_rank(
        all_rankings,
        RELEVANT_INDICES
    )

    recall_1 = mean_recall_at_k(
        all_rankings,
        RELEVANT_INDICES,
        k=1
    )

    recall_3 = mean_recall_at_k(
        all_rankings,
        RELEVANT_INDICES,
        k=3
    )

    print(f"MRR:      {mrr:.4f}")
    print(f"Recall@1: {recall_1:.4f}")
    print(f"Recall@3: {recall_3:.4f}")


if __name__ == "__main__":
    main()