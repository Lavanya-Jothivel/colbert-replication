import json

from src.data import ColBERTTripletDataset


def test_load_jsonl(tmp_path):

    path = tmp_path / "sample.jsonl"

    examples = [
        {
            "query": "what is python?",
            "positive": "Python is a programming language.",
            "negative": "Paris is in France."
        },
        {
            "query": "what is machine learning?",
            "positive": "Machine learning learns from data.",
            "negative": "Basketball is a sport."
        }
    ]

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        for example in examples:
            file.write(
                json.dumps(example) + "\n"
            )

    dataset = ColBERTTripletDataset.from_jsonl(
        path
    )

    assert len(dataset) == 2

    assert (
        dataset[0]["query"]
        == "what is python?"
    )