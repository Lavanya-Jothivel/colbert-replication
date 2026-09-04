import torch
import torch.nn.functional as F


def pairwise_ranking_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor
) -> torch.Tensor:
    """
    Pairwise ranking loss.

    Encourages:
        positive_score > negative_score

    Loss:
        -log(sigmoid(pos - neg))
    """

    score_difference = positive_scores - negative_scores

    loss = -F.logsigmoid(score_difference)

    return loss.mean()