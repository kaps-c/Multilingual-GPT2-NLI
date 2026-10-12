"""Recovered fertility, representation, attention and confidence analysis."""

import numpy as np
import torch


def calibration_metrics(probabilities, gold, bins=10):
    """Multiclass Brier score and equal-width expected calibration error."""
    probabilities = np.asarray(probabilities, dtype=float)
    gold = np.asarray(gold, dtype=int)
    if (
        probabilities.ndim != 2
        or len(probabilities) != len(gold)
        or not len(gold)
    ):
        raise ValueError(
            "probabilities and gold must have matching nonempty rows"
        )
    if bins < 1 or not np.isfinite(probabilities).all():
        raise ValueError("bins must be positive and probabilities finite")
    if (probabilities < 0).any() or not np.allclose(probabilities.sum(1), 1):
        raise ValueError("each probability row must sum to one")
    if (gold < 0).any() or (gold >= probabilities.shape[1]).any():
        raise ValueError("gold label outside probability columns")
    confidence = probabilities.max(1)
    correct = probabilities.argmax(1) == gold
    assignments = np.minimum((confidence * bins).astype(int), bins - 1)
    reliability = []
    ece = 0.0
    for index in range(bins):
        selected = assignments == index
        count = int(selected.sum())
        if not count:
            continue
        accuracy = float(correct[selected].mean())
        mean_confidence = float(confidence[selected].mean())
        ece += count / len(gold) * abs(accuracy - mean_confidence)
        reliability.append(
            dict(
                lower=index / bins,
                upper=(index + 1) / bins,
                count=count,
                accuracy=accuracy,
                confidence=mean_confidence,
            )
        )
    targets = np.eye(probabilities.shape[1])[gold]
    entropy = -(probabilities * np.log2(np.clip(probabilities, 1e-12, 1))).sum(
        1
    )
    return dict(
        accuracy=float(correct.mean()),
        ece=float(ece),
        brier=float(((probabilities - targets) ** 2).sum(1).mean()),
        mean_entropy_bits=float(entropy.mean()),
        bins=reliability,
    )


def fertility(records, tokenizer):
    """Source definition: subword count / whitespace-estimated word count."""
    words = tokens = 0
    for row in records:
        for text in (row["premise"], row["hypothesis"]):
            words += len(text.split())
            tokens += len(tokenizer.encode(text, add_special_tokens=False))
    if not words:
        raise ValueError(
            "fertility needs at least one whitespace-delimited word"
        )
    return {
        "fertility": tokens / words,
        "words": words,
        "tokens": tokens,
        "count": len(records),
    }


def correlate_fertility(rows, metric="accuracy"):
    from scipy.stats import pearsonr, spearmanr

    pairs = [
        (r["fertility"], r[metric]) for r in rows if r.get(metric) is not None
    ]
    values = np.array(pairs, dtype=float)
    if len(values) < 3 or any(np.ptp(values[:, i]) == 0 for i in (0, 1)):
        return {
            "count": len(values),
            "pearson": None,
            "spearman": None,
            "reason": "needs three nonconstant paired observations",
        }
    pearson = pearsonr(values[:, 0], values[:, 1])
    spearman = spearmanr(values[:, 0], values[:, 1])
    return {
        "count": len(values),
        "pearson": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


@torch.no_grad()
def collect_embeddings(model, loader, device="cpu"):
    model.eval()
    vectors = []
    for batch in loader:
        hidden = model(
            batch["input_ids"].to(device), batch["attention_mask"].to(device)
        )["last_token"]
        vectors.append(hidden.float().cpu().numpy())
    if not vectors:
        raise ValueError("embedding loader is empty")
    return np.concatenate(vectors)


def embedding_summary(vectors, groups, seed=42):
    from sklearn.manifold import TSNE
    from sklearn.metrics import davies_bouldin_score, silhouette_score
    from sklearn.metrics.pairwise import cosine_distances

    vectors, groups = np.asarray(vectors), np.asarray(groups)
    if len(vectors) != len(groups) or len(vectors) < 4:
        raise ValueError(
            "embedding analysis needs at least four labeled vectors"
        )
    names = sorted(set(groups.tolist()))
    centroids = np.stack([vectors[groups == name].mean(0) for name in names])
    compactness = {
        name: float(
            cosine_distances(vectors[groups == name], centroid[None]).mean()
        )
        for name, centroid in zip(names, centroids, strict=True)
    }
    coordinates = TSNE(
        n_components=2,
        perplexity=min(30, max(1, (len(vectors) - 1) // 3)),
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(vectors)
    result = dict(
        coordinates=coordinates.tolist(),
        groups=groups.tolist(),
        group_names=names,
        centroid_cosine_distances=cosine_distances(centroids).tolist(),
        compactness=compactness,
    )
    if 1 < len(names) < len(vectors):
        result["silhouette"] = float(
            silhouette_score(vectors, groups, metric="cosine")
        )
        result["davies_bouldin"] = float(davies_bouldin_score(vectors, groups))
    return result


@torch.no_grad()
def attention_summary(model, input_ids, attention_mask, layer=-1):
    """Inspect attention maps, entropy and distance for one input."""
    if len(input_ids) != 1:
        raise ValueError("attention inspection accepts one example")
    model.eval()
    weights = model(input_ids, attention_mask, output_attentions=True)[
        "attentions"
    ][layer][0]
    keep = attention_mask[0].bool()
    weights = weights[:, keep][:, :, keep].float()
    positions = torch.arange(weights.shape[-1], device=weights.device)
    distances = (positions[:, None] - positions[None, :]).abs()
    entropy = -(weights * weights.clamp_min(1e-12).log2()).sum(-1).mean(-1)
    mean_distance = (weights * distances).sum(-1).mean(-1)
    return {
        "layer": layer,
        "weights": weights.cpu().tolist(),
        "entropy_bits": entropy.cpu().tolist(),
        "mean_distance": mean_distance.cpu().tolist(),
    }


def error_examples(predictions, limit=5, order="entropy"):
    """Export raw errors with blank fields for manual taxonomy annotation."""
    errors = []
    for row in predictions:
        if row["correct"]:
            continue
        probabilities = np.asarray(row["label_probabilities"])
        entropy = float(
            -(probabilities * np.log2(np.clip(probabilities, 1e-12, 1))).sum()
        )
        errors.append(
            {
                **row,
                "entropy": entropy,
                "confidence": float(probabilities.max()),
                "error_category": "",
                "annotation": "",
            }
        )
    if order not in {"entropy", "confidence"}:
        raise ValueError("order must be entropy or confidence")
    return sorted(errors, key=lambda row: row[order], reverse=True)[:limit]


def premise_attention_mass(weights, hypothesis_start, label_start):
    """Average hypothesis-query attention directed to each prompt segment."""
    weights = np.asarray(weights)
    if not 0 < hypothesis_start < label_start <= weights.shape[-1]:
        raise ValueError("prompt segment boundaries must be ordered")
    queries = weights[:, hypothesis_start:label_start, :].mean(0)
    total = queries.sum()
    return {
        "hypothesis_to_premise": float(
            queries[:, :hypothesis_start].sum() / total
        ),
        "hypothesis_to_hypothesis": float(
            queries[:, hypothesis_start:label_start].sum() / total
        ),
    }


def prompt_attention_mass(weights, token_ids, tokenizer):
    """Locate segment markers after truncation, including their delimiters."""
    ids = list(token_ids)
    boundaries = []
    for marker in ("\nHypothesis:", "\nLabel:"):
        pattern = tokenizer.encode(marker, add_special_tokens=False)
        matches = [
            index
            for index in range(len(ids) - len(pattern) + 1)
            if ids[index : index + len(pattern)] == pattern
        ]
        if not matches:
            raise ValueError(f"missing prompt delimiter: {marker!r}")
        boundaries.append(matches[-1])
    return premise_attention_mass(weights, *boundaries)


def compare_embeddings(baseline, fine_tuned, languages, seed=42):
    """Compare matching examples in original space and one joint projection."""
    if baseline.shape != fine_tuned.shape or len(languages) != len(baseline):
        raise ValueError("embedding comparisons require matching example rows")
    combined = np.concatenate((baseline, fine_tuned))
    groups = [f"reference/{lang}" for lang in languages] + [
        f"compared/{lang}" for lang in languages
    ]
    result = embedding_summary(combined, groups, seed)
    result["paired_shift_norm"] = np.linalg.norm(
        fine_tuned - baseline, axis=1
    ).tolist()
    return result
