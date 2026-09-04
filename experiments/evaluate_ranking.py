import json

import torch

from src.colbert_model import ColBERTModel
from src.metrics import (
    mean_reciprocal_rank,
    mean_recall_at_k
)


DATA_PATH = "data/msmarco_train_1000.jsonl"

CHECKPOINT_PATH = (
    "experiments/colbert_msmarco_qd_800.pt"
)


def load_examples():
    examples = []

    with open(
        DATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        for line in file:
            examples.append(
                json.loads(line)
            )

    return examples


def main():
    examples = load_examples()

    # Same 200 examples never used for training
    validation_examples = examples[800:]

    model = ColBERTModel(
        model_name="bert-base-uncased",
        projection_dim=128
    )

    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu"
    )

    model.load_state_dict(state_dict)
    model.eval()

    # Build a 20-query / 20-document ranking task.
    # Each query has exactly one relevant positive document.
    evaluation_examples = validation_examples[:20]

    queries = [
        example["query"]
        for example in evaluation_examples
    ]

    documents = [
        example["positive"]
        for example in evaluation_examples
    ]

    print("\nRanking evaluation")
    print(f"Queries: {len(queries)}")
    print(f"Candidate documents: {len(documents)}\n")

    all_ranked_indices = []
    relevant_indices = []

    with torch.no_grad():

        scores = model.score_matrix(
            queries,
            documents
        )

        for i in range(len(queries)):

            ranked_indices = torch.argsort(
                scores[i],
                descending=True
            ).tolist()

            all_ranked_indices.append(
                ranked_indices
            )

            # Positive document for query i
            # is document i.
            relevant_indices.append(i)

    mrr = mean_reciprocal_rank(
        all_ranked_indices,
        relevant_indices
    )

    recall_1 = mean_recall_at_k(
        all_ranked_indices,
        relevant_indices,
        1
    )

    recall_3 = mean_recall_at_k(
        all_ranked_indices,
        relevant_indices,
        3
    )

    recall_5 = mean_recall_at_k(
        all_ranked_indices,
        relevant_indices,
        5
    )

    print(f"MRR:      {mrr:.4f}")
    print(f"Recall@1: {recall_1:.4f}")
    print(f"Recall@3: {recall_3:.4f}")
    print(f"Recall@5: {recall_5:.4f}")

    print("\nExample rankings:\n")

    for i in range(min(5, len(queries))):

        rank = (
            all_ranked_indices[i].index(i)
            + 1
        )

        print(
            f"Query {i + 1}: "
            f"relevant document rank = {rank}"
        )


if __name__ == "__main__":
    main()