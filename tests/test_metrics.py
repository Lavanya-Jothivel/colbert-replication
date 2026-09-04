from src.metrics import (
    reciprocal_rank,
    mean_reciprocal_rank,
    recall_at_k,
    mean_recall_at_k
)


def test_reciprocal_rank():
    ranking = [2, 0, 1]

    rr = reciprocal_rank(
        ranking,
        relevant_index=0
    )

    assert rr == 0.5


def test_mean_reciprocal_rank():
    rankings = [
        [0, 1, 2],
        [2, 1, 0]
    ]

    relevant = [
        0,
        1
    ]

    mrr = mean_reciprocal_rank(
        rankings,
        relevant
    )

    assert mrr == 0.75


def test_recall_at_k():
    ranking = [2, 1, 0]

    assert recall_at_k(
        ranking,
        relevant_index=1,
        k=2
    ) == 1.0

    assert recall_at_k(
        ranking,
        relevant_index=0,
        k=2
    ) == 0.0


def test_mean_recall_at_k():
    rankings = [
        [0, 1, 2],
        [2, 1, 0]
    ]

    relevant = [
        0,
        0
    ]

    recall = mean_recall_at_k(
        rankings,
        relevant,
        k=2
    )

    assert recall == 0.5