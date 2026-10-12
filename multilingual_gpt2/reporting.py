"""Result presentation shared by the thin notebook and exported figures."""

import json
from pathlib import Path


def load_result(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_summary(results):
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "training examples": result["settings"]["train_count"],
                "test examples": result["test"]["count"],
                "test accuracy (%)": round(
                    result["test"]["accuracy"] * 100, 2
                ),
                "majority baseline (%)": round(
                    result["test"]["majority_baseline"] * 100, 2
                ),
                "valid labels (%)": round(result["test"]["coverage"] * 100, 2),
                "selected epoch": result["selected_epoch"],
            }
            for result in results
        ]
    )


def plot_learning_curve(result):
    import matplotlib.pyplot as plt
    import pandas as pd

    losses = pd.Series(result["losses"])
    figure, axis = plt.subplots(figsize=(9, 3.5))
    axis.plot(
        losses.index + 1,
        losses,
        alpha=0.2,
        color="#2563eb",
        label="batch loss",
    )
    axis.plot(
        losses.index + 1,
        losses.rolling(25, min_periods=1).mean(),
        color="#2563eb",
        label="25-batch mean",
    )
    pretraining = result["settings"].get("command") == "pretrain"
    axis.set(
        xlabel="Training batch",
        ylabel="Next-token cross-entropy"
        if pretraining
        else "Next-label cross-entropy",
        title="Toy pretraining" if pretraining else "English NLI fine-tuning",
    )
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def plot_confusion(result):
    """Count every held-out prediction, including invalid generated labels."""
    import matplotlib.pyplot as plt
    import numpy as np

    counts = np.zeros((3, 4), dtype=int)
    for row in result["test"]["predictions"]:
        predicted = (
            int(row["prediction"])
            if row["prediction"] in {"0", "1", "2"}
            else 3
        )
        counts[int(row["gold"]), predicted] += 1
    labels = ["entailment", "contradiction", "neutral"]
    figure, axis = plt.subplots(figsize=(6, 4))
    figure.colorbar(axis.imshow(counts, cmap="Blues"), ax=axis)
    for (row, column), count in np.ndenumerate(counts):
        axis.text(
            column,
            row,
            str(count),
            ha="center",
            va="center",
            color="white" if count > counts.max() / 2 else "black",
        )
    axis.set(
        xticks=range(4),
        xticklabels=labels + ["invalid"],
        yticks=range(3),
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="Gold label",
        title="English held-out confusion matrix",
    )
    figure.tight_layout()
    return figure


def prediction_examples(result, count=6):
    import pandas as pd

    names = {"0": "entailment", "1": "contradiction", "2": "neutral"}
    return pd.DataFrame(
        [
            {
                "premise": row["premise"],
                "hypothesis": row["hypothesis"],
                "gold": names[row["gold"]],
                "prediction": names.get(row["prediction"], "invalid"),
                "correct": row["correct"],
            }
            for row in result["test"]["predictions"][:count]
        ]
    )
