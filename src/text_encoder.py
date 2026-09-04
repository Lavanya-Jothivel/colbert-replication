import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        projection_dim: int = 128
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        # Add ColBERT-style special markers
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": [
                "[Q]",
                "[D]"
            ]
        })

        self.bert = AutoModel.from_pretrained(
            model_name
        )

        # Resize BERT embeddings because we added tokens
        self.bert.resize_token_embeddings(
            len(self.tokenizer)
        )

        hidden_size = self.bert.config.hidden_size

        self.projection = nn.Linear(
            hidden_size,
            projection_dim,
            bias=False
        )

    def _encode(self, texts, marker):
        marked_texts = [
            f"{marker} {text}"
            for text in texts
        ]

        encoded = self.tokenizer(
            marked_texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        device = next(self.parameters()).device

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        outputs = self.bert(**encoded)

        token_embeddings = (
            outputs.last_hidden_state
        )

        projected = self.projection(
            token_embeddings
        )

        normalized = F.normalize(
            projected,
            p=2,
            dim=-1
        )

        return (
            normalized,
            encoded["attention_mask"]
        )

    def encode_queries(self, texts):
        return self._encode(
            texts,
            "[Q]"
        )

    def encode_documents(self, texts):
        return self._encode(
            texts,
            "[D]"
        )

    # Keep old behaviour for compatibility
    def forward(self, texts):
        return self.encode_queries(texts)