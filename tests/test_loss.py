import torch

from src.loss import pairwise_ranking_loss


def test_loss_is_scalar():
    positive = torch.tensor([2.0, 3.0])
    negative = torch.tensor([1.0, 1.5])

    loss = pairwise_ranking_loss(
        positive,
        negative
    )

    assert loss.ndim == 0


def test_better_ranking_has_lower_loss():
    good_positive = torch.tensor([3.0])
    good_negative = torch.tensor([1.0])

    bad_positive = torch.tensor([1.0])
    bad_negative = torch.tensor([3.0])

    good_loss = pairwise_ranking_loss(
        good_positive,
        good_negative
    )

    bad_loss = pairwise_ranking_loss(
        bad_positive,
        bad_negative
    )

    assert good_loss < bad_loss