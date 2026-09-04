from src.colbert_model import ColBERTModel


def main():
    model = ColBERTModel(
        model_name="bert-base-uncased",
        projection_dim=128
    )

    query = "what is machine learning?"

    documents = [
        "Machine learning is a field of artificial intelligence that learns patterns from data.",
        "The Eiffel Tower is located in Paris, France.",
        "Python is a popular programming language used for software development.",
        "Deep learning uses neural networks with many layers."
    ]

    results = []

    for document in documents:
        score = model.score(query, document)

        results.append(
            (
                float(score.detach().cpu()),
                document
            )
        )

    results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    print("\nQuery:")
    print(query)

    print("\nRanking:")

    for rank, (score, document) in enumerate(results, start=1):
        print(
            f"{rank}. Score: {score:.4f}\n"
            f"   {document}\n"
        )


if __name__ == "__main__":
    main()