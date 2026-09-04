from src.colbert_model import ColBERTModel
from src.indexer import ColBERTIndex


def test_save_and_load_index(tmp_path):
    model = ColBERTModel(
        model_name="bert-base-uncased",
        projection_dim=128
    )

    documents = [
        "Python is a programming language.",
        "Machine learning learns from data."
    ]

    index = ColBERTIndex()
    index.build(model, documents)

    path = tmp_path / "index.pt"

    index.save(path)

    loaded_index = ColBERTIndex()
    loaded_index.load(path)

    assert len(loaded_index) == 2
    assert loaded_index.documents == documents
    assert len(loaded_index.embeddings) == 2
    assert len(loaded_index.masks) == 2