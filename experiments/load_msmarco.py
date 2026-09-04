from datasets import load_dataset


def main():
    print("Loading MS MARCO triplets...")

    dataset = load_dataset(
        "sentence-transformers/msmarco-msmarco-MiniLM-L6-v3",
        "triplet",
        split="train"
    )

    print(f"\nTotal examples: {len(dataset)}")

    # Keep only a small subset for local CPU experiments
    small_dataset = dataset.select(range(100))

    print(f"Using subset: {len(small_dataset)} examples")

    print("\nFirst example:\n")

    example = small_dataset[0]

    print("QUERY:")
    print(example["query"])

    print("\nPOSITIVE:")
    print(example["positive"])

    print("\nNEGATIVE:")
    print(example["negative"])


if __name__ == "__main__":
    main()