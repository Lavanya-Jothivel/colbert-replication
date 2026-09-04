import torch
import torch.nn.functional as F


def inbatch_cross_entropy_loss(scores: torch.Tensor) -> torch.Tensor:
    """
    scores: [batch_size, batch_size]

    scores[i, j] = score of query i with document j.
    Correct document for query i is document i.
    """

    batch_size = scores.size(0)

    targets = torch.arange(
        batch_size,
        device=scores.device
    )

    return F.cross_entropy(scores, targets)