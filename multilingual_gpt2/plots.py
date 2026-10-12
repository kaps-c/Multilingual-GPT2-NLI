"""Export compact figures from saved analysis artifacts."""

import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path("artifacts/matplotlib").resolve())
)

import matplotlib.pyplot as plt
import numpy as np

from .reporting import load_result


def diagnostic_figures(directory, languages=None):
    """Yield named figures for notebooks and standalone exports."""
    directory = Path(directory)
    report = load_result(directory / "summary.json")
    rows = report["languages"]
    labels = [row["language"] for row in rows]
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.bar(labels, [row["accuracy"] * 100 for row in rows], color="#2563eb")
    axis.set(
        ylabel="Held-out accuracy (%)", title="Cross-lingual NLI transfer"
    )
    figure.tight_layout()
    yield "accuracy", figure

    figure, axis = plt.subplots(figsize=(6, 4))
    x = np.array([row["fertility"] for row in rows])
    y = np.array([row["accuracy"] * 100 for row in rows])
    axis.scatter(x, y)
    for label, left, right in zip(labels, x, y, strict=True):
        axis.annotate(
            label, (left, right), xytext=(4, 4), textcoords="offset points"
        )
    if len(x) > 2 and np.ptp(x):
        axis.plot(
            np.sort(x), np.polyval(np.polyfit(x, y, 1), np.sort(x)), alpha=0.5
        )
    axis.set(
        xlabel="Subwords / whitespace-estimated word", ylabel="Accuracy (%)"
    )
    figure.tight_layout()
    yield "fertility", figure

    embeddings = load_result(directory / "embedding-summary.json")
    coordinates = np.array(embeddings["coordinates"])
    groups = np.array(embeddings["groups"])
    figure, axis = plt.subplots(figsize=(8, 5))
    for label in sorted(set(groups)):
        points = coordinates[groups == label]
        axis.scatter(points[:, 0], points[:, 1], s=8, label=label, alpha=0.7)
    axis.legend(ncol=3, fontsize=8)
    axis.set(
        title="Multilingual sentence embeddings (t-SNE)",
        xlabel="Dimension 1",
        ylabel="Dimension 2",
    )
    figure.tight_layout()
    yield "embeddings", figure

    figure, axis = plt.subplots(figsize=(6, 5))
    values = np.array(embeddings["centroid_cosine_distances"])
    figure.colorbar(axis.imshow(values), ax=axis)
    names = embeddings["group_names"]
    axis.set(
        xticks=range(len(names)),
        yticks=range(len(names)),
        xticklabels=names,
        yticklabels=names,
        title="Centroid cosine distances",
    )
    figure.tight_layout()
    yield "centroids", figure

    for index, comparison in enumerate(report.get("comparisons", [])):
        figure, axis = plt.subplots(figsize=(8, 4))
        positions = np.arange(len(labels))
        axis.bar(positions - 0.2, y, 0.4, label="reference checkpoint")
        axis.bar(
            positions + 0.2,
            [row["accuracy"] * 100 for row in comparison["languages"]],
            0.4,
            label="compared checkpoint",
        )
        axis.set(
            xticks=positions,
            xticklabels=labels,
            ylim=(0, 100),
            ylabel="Accuracy (%)",
            title="Same-example checkpoint comparison",
        )
        axis.legend()
        figure.tight_layout()
        yield f"comparison-{index}", figure

        paired = load_result(directory / f"comparison-{index}-embeddings.json")
        coordinates = np.asarray(paired["coordinates"])
        count = len(coordinates) // 2
        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        for label in sorted(set(groups)):
            selected = groups == label
            reference_points = axes[0].scatter(
                *coordinates[:count][selected].T,
                s=18,
                label=f"reference/{label}",
            )
            axes[0].scatter(
                *coordinates[count:][selected].T,
                marker="x",
                s=18,
                color=reference_points.get_facecolor()[0],
                label=f"compared/{label}",
            )
        axes[0].set(
            title="Shared t-SNE projection",
            xlabel="Dimension 1",
            ylabel="Dimension 2",
        )
        axes[0].legend(fontsize=6, ncol=2)
        axes[1].hist(paired["paired_shift_norm"], bins=10)
        axes[1].set(
            title="Paired embedding shifts",
            xlabel="Euclidean distance",
            ylabel="Example count",
        )
        figure.tight_layout()
        yield f"comparison-{index}-embeddings", figure

    for label in labels if languages is None else languages:
        evaluation = load_result(directory / f"{label}-evaluation.json")
        bins = evaluation["calibration_conditional_on_labels"]["bins"]
        figure, axis = plt.subplots(figsize=(4, 4))
        axis.plot([0, 1], [0, 1], "--", color="gray")
        axis.plot(
            [b["confidence"] for b in bins],
            [b["accuracy"] for b in bins],
            "o-",
        )
        axis.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="Conditional label confidence",
            ylabel="Accuracy",
            title=f"Calibration: {label}",
        )
        figure.tight_layout()
        yield f"{label}-calibration", figure
        figure, axes = plt.subplots(2, 2, figsize=(10, 8))
        for column, (layer, name) in enumerate(((0, "shallow"), (-1, "deep"))):
            with np.load(directory / f"{label}-attention-{layer}.npz") as data:
                attention = data["weights"]
            axis = axes[0, column]
            figure.colorbar(axis.imshow(attention[0], aspect="auto"), ax=axis)
            axis.set(
                xlabel="Key token position",
                ylabel="Query token position",
                title=f"{label}: {name} layer, head 0",
            )
            summary = load_result(
                directory / f"{label}-attention-{layer}.json"
            )
            for axis, metric in zip(
                axes[1], ("entropy_bits", "mean_distance")
            ):
                axis.plot(summary[metric], "o-", label=name)
        for axis, title in zip(
            axes[1], ("Entropy (bits)", "Attention distance")
        ):
            axis.set(xlabel="Head index", ylabel=title)
            axis.legend()
        figure.tight_layout()
        yield f"{label}-attention", figure

        probabilities = np.asarray(
            [row["label_probabilities"] for row in evaluation["predictions"]]
        )
        confidence = probabilities.max(1)
        entropy = -(
            probabilities * np.log2(np.clip(probabilities, 1e-12, 1))
        ).sum(1)
        correct = np.asarray(
            [row["correct"] for row in evaluation["predictions"]]
        )
        figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        for axis, values, name in zip(
            axes,
            (confidence, entropy),
            ("Conditional confidence", "Entropy (bits)"),
        ):
            edges = np.linspace(
                0, 1 if name.startswith("Conditional") else np.log2(3), 9
            )
            for mask, label_name in (
                (correct, "correct"),
                (~correct, "incorrect"),
            ):
                axis.hist(
                    values[mask], bins=edges, alpha=0.6, label=label_name
                )
            axis.set(xlabel=name, ylabel="Example count")
            axis.legend()
        figure.suptitle(f"Prediction uncertainty: {label}")
        figure.tight_layout()
        yield f"{label}-uncertainty", figure


def render_diagnostics(directory):
    """Export the same figures used by the notebook without opening windows."""
    plt.switch_backend("Agg")
    for name, figure in diagnostic_figures(directory):
        figure.savefig(Path(directory) / f"{name}.png", dpi=140)
        plt.close(figure)
