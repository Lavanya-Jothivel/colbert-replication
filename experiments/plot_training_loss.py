import csv

import matplotlib.pyplot as plt


CSV_PATH = "experiments/msmarco_training_loss.csv"
OUTPUT_PATH = "experiments/msmarco_training_loss.png"


def main():
    epochs = []
    losses = []

    with open(
        CSV_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            epochs.append(
                int(row["epoch"])
            )

            losses.append(
                float(row["loss"])
            )

    plt.figure(figsize=(7, 5))

    plt.plot(
        epochs,
        losses,
        marker="o"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Average Ranking Loss")

    plt.title(
        "ColBERT Training Loss on MS MARCO Subset"
    )

    plt.xticks(epochs)

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PATH,
        dpi=200
    )

    print(
        f"Saved plot to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()