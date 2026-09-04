import torch

from src.colbert_model import ColBERTModel


def test_score_matrix_shape():
    model = ColBERTModel(
        model_name="bert-base-uncased",
        projection_dim=128
    )

    queries = [
        "machine learning",
        "python programming"
    ]

    documents = [
        "Machine learning learns patterns from data.",
        "Python is a programming language.",
        "Paris is the capital of France."
    ]

    scores = model.score_matrix(
        queries,
        documents
    )

    assert scores.shape == (2, 3)


def test_score_matrix_is_finite():
    model = ColBERTModel(
        model_name="bert-base-uncased",
        projection_dim=128
    )

    scores = model.score_matrix(
        ["machine learning"],
        [
            "Machine learning uses data.",
            "The Eiffel Tower is in Paris."
        ]
    )

    assert torch.isfinite(scores).all()