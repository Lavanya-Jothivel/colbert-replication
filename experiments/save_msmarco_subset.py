import json
import os

from datasets import load_dataset


OUTPUT_PATH = "data/msmarco_train_100.jsonl"


def main():
    print("Loading MS MARCO triplets...")

    dataset = load_dataset(
        "sentence-transformers/msmarco-msmarco-MiniLM-L6-v3",
        "triplet",
        split="train"
    )

    subset = dataset.select(range(100))

    os.makedirs(
        "data",
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        for example in subset:

            record = {
                "query": example["query"],
                "positive": example["positive"],
                "negative": example["negative"]
            }

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )

    print(
        f"\nSaved {len(subset)} examples to:"
    )

    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()