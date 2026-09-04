import torch

from src.maxsim import maxsim_score


def colbert_score(query_embeddings: torch.Tensor,
                  document_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute the ColBERT late-interaction score between a query
    and a document.

    Args:
        query_embeddings:
            Tensor of shape [query_length, embedding_dim]

        document_embeddings:
            Tensor of shape [document_length, embedding_dim]

    Returns:
        Scalar tensor containing the ColBERT score.
    """

    return maxsim_score(query_embeddings, document_embeddings)