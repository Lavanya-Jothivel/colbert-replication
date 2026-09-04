from src.colbert_model import ColBERTModel
from src.indexer import ColBERTIndex


def test_search_returns_top_k():
    model = ColBERTModel(
        model_name="bert-base-uncased",
        projection_dim=128
    )

    documents = [
        "Python is a programming language.",
        "Machine learning learns patterns from data.",
        "Paris is the capital of France."
    ]

    index = ColBERTIndex()
    index.build(model, documents)

    results = index.search(
        model,
        "python programming",
        top_k=2
    )

    assert len(results) == 2


def test_search_result_structure():
    model = ColBERTModel(
        model_name="bert-base-uncased",
        projection_dim=128
    )

    index = ColBERTIndex()

    index.build(
        model,
        [
            "Python is used in data science.",
            "The Eiffel Tower is in Paris."
        ]
    )

    results = index.search(
        model,
        "python",
        top_k=1
    )

    result = results[0]

    assert "document" in result
    assert "score" in result
    assert "index" in result