import torch

from src.batch_loss import inbatch_cross_entropy_loss


def test_inbatch_loss_is_scalar():
    scores = torch.tensor([
        [3.0, 1.0],
        [1.0, 3.0]
    ])

    loss = inbatch_cross_entropy_loss(scores)

    assert loss.ndim == 0


def test_good_diagonal_has_lower_loss():
    good_scores = torch.tensor([
        [4.0, 1.0],
        [1.0, 4.0]
    ])

    bad_scores = torch.tensor([
        [1.0, 4.0],
        [4.0, 1.0]
    ])

    good_loss = inbatch_cross_entropy_loss(good_scores)
    bad_loss = inbatch_cross_entropy_loss(bad_scores)

    assert good_loss < bad_loss