import torch
import torch.nn as nn

from src.text_encoder import TextEncoder
from src.maxsim import maxsim_score


class ColBERTModel(nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        projection_dim: int = 128
    ):
        super().__init__()

        self.encoder = TextEncoder(
            model_name=model_name,
            projection_dim=projection_dim
        )

    def encode_queries(self, texts):
        return self.encoder.encode_queries(
            texts
        )

    def encode_documents(self, texts):
        return self.encoder.encode_documents(
            texts
        )

    # Compatibility helper
    def encode(self, texts):
        return self.encode_queries(texts)

    def score(
        self,
        query: str,
        document: str
    ):
        query_embeddings, query_mask = (
            self.encode_queries([query])
        )

        document_embeddings, document_mask = (
            self.encode_documents([document])
        )

        query_embeddings = query_embeddings[0]
        document_embeddings = document_embeddings[0]

        query_mask = query_mask[0].bool()
        document_mask = document_mask[0].bool()

        query_embeddings = query_embeddings[
            query_mask
        ]

        document_embeddings = document_embeddings[
            document_mask
        ]

        return maxsim_score(
            query_embeddings,
            document_embeddings
        )

    def score_matrix(
        self,
        queries,
        documents
    ):
        query_embeddings, query_masks = (
            self.encode_queries(queries)
        )

        document_embeddings, document_masks = (
            self.encode_documents(documents)
        )

        rows = []

        for i in range(len(queries)):
            q = query_embeddings[i][
                query_masks[i].bool()
            ]

            row_scores = []

            for j in range(len(documents)):
                d = document_embeddings[j][
                    document_masks[j].bool()
                ]

                score = maxsim_score(
                    q,
                    d
                )

                row_scores.append(score)

            rows.append(
                torch.stack(row_scores)
            )

        return torch.stack(rows)