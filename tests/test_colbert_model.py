import torch

from src.colbert_model import ColBERTModel


def test_colbert_model_returns_scalar():
    model = ColBERTModel(
        model_name="bert-base-uncased",
        projection_dim=128
    )

    score = model.score(
        "what is artificial intelligence?",
        "Artificial intelligence is the field of building intelligent machines."
    )

    assert isinstance(score, torch.Tensor)
    assert score.ndim == 0


def test_colbert_model_score_is_finite():
    model = ColBERTModel(
        model_name="bert-base-uncased",
        projection_dim=128
    )

    score = model.score(
        "machine learning",
        "Machine learning allows computers to learn patterns from data."
    )

    assert torch.isfinite(score)