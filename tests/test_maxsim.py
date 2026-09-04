import torch

from src.maxsim import maxsim_score


def test_maxsim_returns_scalar():
    query = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    document = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])

    score = maxsim_score(query, document)

    assert score.ndim == 0


def test_perfect_matches_score_two():
    query = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    document = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    score = maxsim_score(query, document)

    assert torch.isclose(score, torch.tensor(2.0))