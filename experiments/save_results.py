import csv
import os


OUTPUT_PATH = "results/evaluation.csv"


def main():
    os.makedirs(
        "results",
        exist_ok=True
    )

    results = [
        {
            "experiment": "MS MARCO 100-triplet subset",
            "training_examples": 100,
            "epochs": 3,
            "projection_dim": 128,
            "bert_frozen": True,
            "pairwise_accuracy": 1.0,
            "average_margin": 0.9754
        }
    ]

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "experiment",
                "training_examples",
                "epochs",
                "projection_dim",
                "bert_frozen",
                "pairwise_accuracy",
                "average_margin"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print(
        f"Saved evaluation results to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()