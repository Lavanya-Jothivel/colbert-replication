from src.colbert_model import ColBERTModel
from src.indexer import ColBERTIndex


def test_index_build():
    model = ColBERTModel(
        model_name="bert-base-uncased",
        projection_dim=128
    )

    documents = [
        "Machine learning learns from data.",
        "Python is a programming language."
    ]

    index = ColBERTIndex()

    index.build(
        model,
        documents
    )

    assert len(index) == 2
    assert len(index.embeddings) == 2
    assert len(index.masks) == 2


def test_index_embedding_dimension():
    model = ColBERTModel(
        model_name="bert-base-uncased",
        projection_dim=128
    )

    index = ColBERTIndex()

    index.build(
        model,
        ["Artificial intelligence is useful."]
    )

    assert index.embeddings[0].shape[-1] == 128