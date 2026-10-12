"""Reproduce multilingual evaluation and the PDF's diagnostic analyses."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from .analysis import (
    attention_summary,
    calibration_metrics,
    collect_embeddings,
    compare_embeddings,
    correlate_fertility,
    embedding_summary,
    error_examples,
    fertility,
    prompt_attention_mass,
)
from .cli import load_tokenizer, make_loader, write_json
from .data import LANGUAGES, NLIDataset, read_records
from .evaluation import evaluate_nli
from .training import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--compare-checkpoints", nargs="*", default=[])
    parser.add_argument("--model", default="artifacts/gpt2")
    parser.add_argument("--data", default="data/xnli")
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGES))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--embedding-limit", type=int, default=200)
    parser.add_argument("--output", default="artifacts/analysis")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args.model)
    model, metadata = load_checkpoint(args.checkpoint, args.device)
    rows, all_errors, embeddings, groups = [], [], [], []
    for language in args.languages:
        records = read_records(
            Path(args.data) / language / "test.jsonl",
            language,
            args.limit,
            args.seed,
        )
        loader = make_loader(NLIDataset(records, tokenizer, 512), tokenizer, 4)
        evaluation = evaluate_nli(model, tokenizer, loader, args.device)
        predictions = evaluation.pop("predictions")
        calibration = calibration_metrics(
            [row["label_probabilities"] for row in predictions],
            [int(row["gold"]) for row in predictions],
        )
        # calibration conditions on digits; generation can still be invalid.
        write_json(
            output / f"{language}-evaluation.json",
            {
                **evaluation,
                "calibration_conditional_on_labels": calibration,
                "predictions": predictions,
            },
        )
        fertility_path = Path(args.data) / language / "train.jsonl"
        fertility_split = "train"
        if not fertility_path.exists():
            fertility_path = Path(args.data) / language / "dev.jsonl"
            fertility_split = "dev"
        fertility_rows = read_records(
            fertility_path, language, min(args.limit or 3927, 3927), args.seed
        )
        rows.append(
            {
                "language": language,
                **evaluation,
                **{
                    key: value
                    for key, value in fertility(
                        fertility_rows, tokenizer
                    ).items()
                    if key != "count"
                },
                "fertility_count": len(fertility_rows),
                "fertility_split": fertility_split,
                "ece": calibration["ece"],
                "brier": calibration["brier"],
                "entropy": calibration["mean_entropy_bits"],
            }
        )
        all_errors.extend(error_examples(predictions))
        dev = read_records(
            Path(args.data) / language / "dev.jsonl",
            language,
            args.embedding_limit,
            args.seed,
        )
        dataset = NLIDataset(dev, tokenizer, 256)
        vectors = collect_embeddings(
            model, make_loader(dataset, tokenizer, 4), args.device
        )
        embeddings.append(vectors)
        groups.extend([language] * len(vectors))
        example = dataset[0]
        for layer in (0, -1):
            attention = attention_summary(
                model,
                example["input_ids"][None].to(args.device),
                example["attention_mask"][None].to(args.device),
                layer,
            )
            weights = np.asarray(attention.pop("weights"))
            attention.update(
                prompt_attention_mass(
                    weights, example["input_ids"].tolist(), tokenizer
                )
            )
            np.savez_compressed(
                output / f"{language}-attention-{layer}.npz",
                weights=weights,
                tokens=example["input_ids"].numpy(),
            )
            write_json(
                output / f"{language}-attention-{layer}.json", attention
            )
        print(f"analyzed {language}: {evaluation['accuracy']:.3f}", flush=True)
    vectors = np.concatenate(embeddings)
    np.savez_compressed(
        output / "embeddings.npz", vectors=vectors, groups=np.asarray(groups)
    )
    write_json(
        output / "embedding-summary.json",
        embedding_summary(vectors, groups, args.seed),
    )
    report = {
        "settings": vars(args),
        "checkpoint_metadata": metadata,
        "languages": rows,
        "note": (
            "fertility uses whitespace word counts; calibration is "
            "conditional on NLI labels"
        ),
    }
    if len(rows) >= 3:
        report["fertility_accuracy"] = correlate_fertility(rows)
        report["fertility_entropy"] = correlate_fertility(rows, "entropy")
    report["accuracy_ranking"] = [
        row["language"]
        for row in sorted(rows, key=lambda row: row["accuracy"], reverse=True)
    ]
    families = {
        "en": "Indo-European",
        "bg": "Indo-European",
        "de": "Indo-European",
        "el": "Indo-European",
        "es": "Indo-European",
        "fr": "Indo-European",
        "hi": "Indo-European",
        "ru": "Indo-European",
        "ur": "Indo-European",
        "ar": "Afro-Asiatic",
        "sw": "Niger-Congo",
        "th": "Kra-Dai",
        "tr": "Turkic",
        "vi": "Austroasiatic",
        "zh": "Sino-Tibetan",
    }
    report["families"] = {
        family: {
            "languages": [
                r["language"]
                for r in rows
                if families[r["language"]] == family
            ],
            "mean_accuracy": float(
                np.mean(
                    [
                        r["accuracy"]
                        for r in rows
                        if families[r["language"]] == family
                    ]
                )
            ),
        }
        for family in sorted({families[r["language"]] for r in rows})
    }
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()
    report["comparisons"] = []
    for index, checkpoint in enumerate(args.compare_checkpoints):
        compared, _ = load_checkpoint(checkpoint, args.device)
        comparison_rows, comparison_vectors = [], []
        for baseline in rows:
            language = baseline["language"]
            test = read_records(
                Path(args.data) / language / "test.jsonl",
                language,
                args.limit,
                args.seed,
            )
            evaluation = evaluate_nli(
                compared,
                tokenizer,
                make_loader(NLIDataset(test, tokenizer, 512), tokenizer, 4),
                args.device,
            )
            comparison_rows.append(
                {
                    "language": language,
                    "accuracy": evaluation["accuracy"],
                    "delta_accuracy": evaluation["accuracy"]
                    - baseline["accuracy"],
                    "fertility": baseline["fertility"],
                }
            )
            dev = read_records(
                Path(args.data) / language / "dev.jsonl",
                language,
                args.embedding_limit,
                args.seed,
            )
            dataset = NLIDataset(dev, tokenizer, 256)
            comparison_vectors.append(
                collect_embeddings(
                    compared, make_loader(dataset, tokenizer, 4), args.device
                )
            )
            example = dataset[0]
            for layer in (0, -1):
                attention = attention_summary(
                    compared,
                    example["input_ids"][None].to(args.device),
                    example["attention_mask"][None].to(args.device),
                    layer,
                )
                weights = np.asarray(attention.pop("weights"))
                attention.update(
                    prompt_attention_mass(
                        weights, example["input_ids"].tolist(), tokenizer
                    )
                )
                write_json(
                    output / f"comparison-{index}-{language}-{layer}.json",
                    attention,
                )
        write_json(
            output / f"comparison-{index}-embeddings.json",
            compare_embeddings(
                vectors, np.concatenate(comparison_vectors), groups, args.seed
            ),
        )
        report["comparisons"].append(
            {
                "checkpoint": checkpoint,
                "languages": comparison_rows,
                "fertility_gain": correlate_fertility(
                    comparison_rows, "delta_accuracy"
                ),
            }
        )
        del compared
        if args.device == "cuda":
            torch.cuda.empty_cache()
    write_json(output / "summary.json", report)
    if all_errors:
        with (output / "errors-for-annotation.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(all_errors[0]))
            writer.writeheader()
            writer.writerows(all_errors)
    from .plots import render_diagnostics

    render_diagnostics(output)


if __name__ == "__main__":
    main()
