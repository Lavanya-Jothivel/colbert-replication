import csv
import os

import matplotlib.pyplot as plt


SMALL_PATH = "results/msmarco_split_metrics.csv"
LARGE_PATH = "results/msmarco_800_200_metrics.csv"

OUTPUT_PATH = "results/training_comparison.png"


def load_metrics(path):
    epochs = []
    losses = []
    accuracies = []

    with open(path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            epochs.append(int(row["epoch"]))
            losses.append(float(row["training_loss"]))
            accuracies.append(
                float(row["validation_accuracy"]) * 100
            )

    return epochs, losses, accuracies


def main():
    os.makedirs("results", exist_ok=True)

    (
        small_epochs,
        small_losses,
        small_accuracies
    ) = load_metrics(SMALL_PATH)

    (
        large_epochs,
        large_losses,
        large_accuracies
    ) = load_metrics(LARGE_PATH)

    # -------------------------
    # Training loss comparison
    # -------------------------

    plt.figure(figsize=(7, 5))

    plt.plot(
        small_epochs,
        small_losses,
        marker="o",
        label="80 training triplets"
    )

    plt.plot(
        large_epochs,
        large_losses,
        marker="o",
        label="800 training triplets"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")

    plt.title(
        "ColBERT Training Loss Comparison"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    loss_path = (
        "results/training_loss_comparison.png"
    )

    plt.savefig(
        loss_path,
        dpi=200
    )

    plt.close()

    # -------------------------
    # Validation comparison
    # -------------------------

    plt.figure(figsize=(7, 5))

    plt.plot(
        small_epochs,
        small_accuracies,
        marker="o",
        label="80 training triplets"
    )

    plt.plot(
        large_epochs,
        large_accuracies,
        marker="o",
        label="800 training triplets"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")

    plt.title(
        "ColBERT Validation Accuracy Comparison"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_PATH,
        dpi=200
    )

    plt.close()

    print(
        "Saved results/training_loss_comparison.png"
    )

    print(
        "Saved results/training_comparison.png"
    )


if __name__ == "__main__":
    main()