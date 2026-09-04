import torch
import torch.nn as nn
import torch.nn.functional as F


class ColBERTEncoder(nn.Module):
    def __init__(self, hidden_size: int, projection_dim: int = 128):
        super().__init__()

        self.projection = nn.Linear(hidden_size, projection_dim, bias=False)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project token embeddings into ColBERT embedding space
        and L2-normalize them.

        Args:
            token_embeddings:
                Tensor of shape
                [batch_size, sequence_length, hidden_size]

        Returns:
            Tensor of shape
            [batch_size, sequence_length, projection_dim]
        """

        projected = self.projection(token_embeddings)

        normalized = F.normalize(
            projected,
            p=2,
            dim=-1
        )

        return normalized