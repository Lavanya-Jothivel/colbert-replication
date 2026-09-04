def reciprocal_rank(ranked_indices, relevant_index):
    """
    Reciprocal Rank for a single query.
    """
    for rank, index in enumerate(ranked_indices, start=1):
        if index == relevant_index:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(all_ranked_indices, relevant_indices):
    """
    Mean Reciprocal Rank across queries.
    """
    scores = []

    for ranked, relevant in zip(
        all_ranked_indices,
        relevant_indices
    ):
        scores.append(
            reciprocal_rank(ranked, relevant)
        )

    return sum(scores) / len(scores)


def recall_at_k(ranked_indices, relevant_index, k):
    """
    Recall@K for a single relevant document.
    """
    top_k = ranked_indices[:k]

    return 1.0 if relevant_index in top_k else 0.0


def mean_recall_at_k(
    all_ranked_indices,
    relevant_indices,
    k
):
    scores = []

    for ranked, relevant in zip(
        all_ranked_indices,
        relevant_indices
    ):
        scores.append(
            recall_at_k(
                ranked,
                relevant,
                k
            )
        )

    return sum(scores) / len(scores)