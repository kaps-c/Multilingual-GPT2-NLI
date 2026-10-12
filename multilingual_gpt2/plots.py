"""Export compact figures from saved analysis artifacts."""

import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path("artifacts/matplotlib").resolve())
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .reporting import load_result


def render_diagnostics(directory):
    directory = Path(directory)
    rows = load_result(directory / "summary.json")["languages"]
    labels = [row["language"] for row in rows]
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.bar(labels, [row["accuracy"] * 100 for row in rows], color="#2563eb")
    axis.set(
        ylabel="Held-out accuracy (%)", title="Cross-lingual NLI transfer"
    )
    figure.tight_layout()
    figure.savefig(directory / "accuracy.png", dpi=140)
    plt.close(figure)

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
    figure.savefig(directory / "fertility.png", dpi=140)
    plt.close(figure)

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
    figure.savefig(directory / "embeddings.png", dpi=140)
    plt.close(figure)

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
    figure.savefig(directory / "centroids.png", dpi=140)
    plt.close(figure)

    for label in labels:
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
        figure.savefig(directory / f"{label}-calibration.png", dpi=140)
        plt.close(figure)
        attention = np.load(directory / f"{label}-attention--1.npz")["weights"]
        figure, axis = plt.subplots(figsize=(5, 4))
        figure.colorbar(axis.imshow(attention[0], aspect="auto"), ax=axis)
        axis.set(
            xlabel="Key token position",
            ylabel="Query token position",
            title=f"Deep-layer attention, head 0: {label}",
        )
        figure.tight_layout()
        figure.savefig(directory / f"{label}-attention.png", dpi=140)
        plt.close(figure)
