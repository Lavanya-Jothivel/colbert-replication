import torch

from src.encoder import ColBERTEncoder


def test_encoder_output_shape():
    encoder = ColBERTEncoder(
        hidden_size=768,
        projection_dim=128
    )

    x = torch.randn(2, 10, 768)

    output = encoder(x)

    assert output.shape == (2, 10, 128)


def test_encoder_normalization():
    encoder = ColBERTEncoder(
        hidden_size=768,
        projection_dim=128
    )

    x = torch.randn(2, 5, 768)

    output = encoder(x)

    norms = torch.linalg.norm(output, dim=-1)

    expected = torch.ones_like(norms)

    assert torch.allclose(
        norms,
        expected,
        atol=1e-5
    )