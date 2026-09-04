import torch

from src.text_encoder import TextEncoder


def test_text_encoder_shape():
    encoder = TextEncoder(
        model_name="bert-base-uncased",
        projection_dim=128
    )

    embeddings, attention_mask = encoder(
        ["what is artificial intelligence?"]
    )

    assert embeddings.ndim == 3

    assert embeddings.shape[0] == 1

    assert embeddings.shape[-1] == 128

    assert attention_mask.shape[:2] == embeddings.shape[:2]


def test_text_encoder_normalization():
    encoder = TextEncoder(
        model_name="bert-base-uncased",
        projection_dim=128
    )

    embeddings, _ = encoder(
        ["machine learning retrieval"]
    )

    norms = torch.linalg.norm(
        embeddings,
        dim=-1
    )

    expected = torch.ones_like(norms)

    assert torch.allclose(
        norms,
        expected,
        atol=1e-5
    )