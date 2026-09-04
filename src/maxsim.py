import torch
import torch.nn.functional as F


def normalize_embeddings(x: torch.Tensor) -> torch.Tensor:
    """
    L2-normalize token embeddings along the last dimension.
    """
    return F.normalize(x, p=2, dim=-1)


def maxsim_score(
    query_embeddings: torch.Tensor,
    document_embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    ColBERT-style MaxSim score.

    query_embeddings:
        [num_query_tokens, embedding_dim]

    document_embeddings:
        [num_document_tokens, embedding_dim]

    Returns:
        scalar tensor
    """

    query_embeddings = normalize_embeddings(query_embeddings)
    document_embeddings = normalize_embeddings(document_embeddings)

    similarity_matrix = query_embeddings @ document_embeddings.T

    max_sim_per_query_token = similarity_matrix.max(dim=1).values

    score = max_sim_per_query_token.sum()

    return score