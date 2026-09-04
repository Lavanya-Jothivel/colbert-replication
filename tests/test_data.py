from src.data import ColBERTTripletDataset


def test_dataset_length():

    examples = [
        {
            "query": "what is machine learning?",
            "positive": "Machine learning learns patterns from data.",
            "negative": "Paris is located in France."
        },
        {
            "query": "what is python?",
            "positive": "Python is a programming language.",
            "negative": "Basketball is a sport."
        }
    ]

    dataset = ColBERTTripletDataset(examples)

    assert len(dataset) == 2


def test_dataset_item():

    examples = [
        {
            "query": "what is python?",
            "positive": "Python is a programming language.",
            "negative": "The Eiffel Tower is in Paris."
        }
    ]

    dataset = ColBERTTripletDataset(examples)

    item = dataset[0]

    assert item["query"] == "what is python?"
    assert item["positive"] == "Python is a programming language."
    assert item["negative"] == "The Eiffel Tower is in Paris."