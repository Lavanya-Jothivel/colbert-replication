import torch

from src.colbert_scorer import colbert_score


def test_colbert_score_returns_scalar():
    query = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0]
    ])

    document = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5]
    ])

    score = colbert_score(query, document)

    assert score.ndim == 0


def test_colbert_score_correct_value():
    query = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0]
    ])

    document = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0]
    ])

    score = colbert_score(query, document)

    assert torch.isclose(score, torch.tensor(2.0))