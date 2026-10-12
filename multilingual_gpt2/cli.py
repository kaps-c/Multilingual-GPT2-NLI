"""Explicit command-line entry points for training and evaluation."""

import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from .config import GPT2Config
from .data import NLIDataset, TextDataset, collate_batch, read_records
from .evaluation import evaluate_nli, generate_gpt2
from .model import GPT2Model
from .optimizer import AdamW
from .training import (
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    train_epoch,
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_tokenizer(path):
    tokenizer = GPT2Tokenizer.from_pretrained(path)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def make_loader(dataset, tokenizer, batch_size, shuffle=False, seed=42):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=partial(collate_batch, pad_token_id=tokenizer.pad_token_id),
    )


def train(args):
    start = time.perf_counter()
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = Path(args.output)
    if (output / "results.json").exists():
        raise FileExistsError(
            "choose a new output directory to preserve prior results"
        )
    tokenizer = load_tokenizer(args.model)
    if args.tiny or args.command == "pretrain":
        model = GPT2Model(
            GPT2Config(
                hidden_size=32 if args.tiny else 128,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=96 if args.tiny else 384,
            )
        )
    else:
        model = GPT2Model.from_pretrained(args.model)
    model.to(args.device)
    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    settings = {
        key: value for key, value in vars(args).items() if key != "handler"
    }
    settings.update(
        initialization="random"
        if args.tiny or args.command == "pretrain"
        else "pretrained",
        python=platform.python_version(),
        torch=str(torch.__version__),
        gpu=torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        label_order=["entailment", "contradiction", "neutral"],
    )
    if args.command == "pretrain":
        dataset = TextDataset(
            args.text, tokenizer, args.max_length, args.train_limit
        )
        dev_loader = test_loader = None
        settings["train_count"] = len(dataset)
    else:
        splits = {}
        for split, limit in (
            ("train", args.train_limit),
            ("dev", args.dev_limit),
            ("test", args.test_limit),
        ):
            rows = []
            for language in args.languages:
                rows.extend(
                    read_records(
                        Path(args.data) / language / f"{split}.jsonl",
                        language,
                        limit,
                        args.seed,
                        args.train_fraction
                        if split == "train"
                        else args.eval_fraction,
                    )
                )
            splits[split] = rows
            settings[f"{split}_count"] = len(rows)
            settings[f"{split}_ids_sha256"] = hashlib.sha256(
                "\n".join(row["id"] for row in rows).encode()
            ).hexdigest()
        dataset = NLIDataset(splits["train"], tokenizer, args.max_length, True)
        dev_loader, test_loader = (
            make_loader(
                NLIDataset(splits[split], tokenizer, args.max_length),
                tokenizer,
                args.batch_size,
            )
            for split in ("dev", "test")
        )
    loader = make_loader(dataset, tokenizer, args.batch_size, True, args.seed)
    print(json.dumps(settings), flush=True)
    baseline = (
        evaluate_nli(model, tokenizer, test_loader, args.device)
        if test_loader
        else None
    )
    losses, epochs = [], []
    best_score = float("-inf")
    for epoch in range(args.epochs):
        epoch_losses = train_epoch(
            model,
            loader,
            optimizer,
            args.device,
            args.accumulation_steps,
            args.mixed_precision,
        )
        losses.extend(epoch_losses)
        score = (
            evaluate_nli(model, tokenizer, dev_loader, args.device)["accuracy"]
            if dev_loader
            else -sum(epoch_losses) / len(epoch_losses)
        )
        epochs.append(
            {
                "epoch": epoch + 1,
                "dev_accuracy": score if dev_loader else None,
                "mean_loss": sum(epoch_losses) / len(epoch_losses),
            }
        )
        if score > best_score:
            best_score = score
            save_checkpoint(
                model,
                args.checkpoint,
                {
                    "settings": settings,
                    "epoch": epoch + 1,
                    "selection_score": score,
                },
            )
        write_json(
            output / "progress.json", {"epochs": epochs, "losses": losses}
        )
        print(json.dumps(epochs[-1]), flush=True)
    del optimizer, model
    if args.device == "cuda":
        torch.cuda.empty_cache()
    model, metadata = load_checkpoint(args.checkpoint, args.device)
    result = {
        "settings": settings,
        "epochs": epochs,
        "losses": losses,
        "selected_epoch": metadata["epoch"],
        "elapsed_seconds": time.perf_counter() - start,
        "checkpoint": args.checkpoint,
        "status": "reproduced_small_run"
        if args.train_limit or args.tiny
        else "reproduced_configured_run",
    }
    if test_loader:
        result["baseline"] = {
            key: value
            for key, value in baseline.items()
            if key != "predictions"
        }
        result["test"] = evaluate_nli(
            model, tokenizer, test_loader, args.device
        )
        counts = Counter(row["label"] for row in splits["test"])
        result["test"]["majority_baseline"] = max(counts.values()) / sum(
            counts.values()
        )
    else:
        result["generation"] = generate_gpt2(
            model, tokenizer, "The", 30, args.device
        )
    write_json(output / "results.json", result)
    print(
        json.dumps(
            {
                "saved": str(output / "results.json"),
                "test_accuracy": result.get("test", {}).get("accuracy"),
            }
        ),
        flush=True,
    )


def evaluate(args):
    tokenizer = load_tokenizer(args.model)
    model, _ = load_checkpoint(args.checkpoint, args.device)
    records = read_records(
        args.data_file, args.language, args.limit, args.seed
    )
    loader = make_loader(
        NLIDataset(records, tokenizer, args.max_length),
        tokenizer,
        args.batch_size,
    )
    result = evaluate_nli(model, tokenizer, loader, args.device)
    write_json(args.output, result)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key != "predictions"
            }
        )
    )


def download(args):
    from datasets import load_dataset

    for language in args.languages:
        dataset = load_dataset(
            "facebook/xnli",
            language,
            cache_dir=str(Path(args.output) / ".cache"),
        )
        for split, rows in dataset.items():
            target = "dev" if split == "validation" else split
            path = Path(args.output) / language / f"{target}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as stream:
                for index, row in enumerate(rows):
                    row.update(
                        label=rows.features["label"].int2str(row["label"]),
                        id=f"{language}/{target}/{index}",
                        language=language,
                    )
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"saved {path}: {len(rows)} examples", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "pretrain"):
        command = commands.add_parser(name)
        command.set_defaults(handler=train)
        command.add_argument("--model", default="artifacts/gpt2")
        command.add_argument("--data", default="data/xnli")
        command.add_argument("--text", default="pretrain.txt")
        command.add_argument("--languages", nargs="+", default=["en"])
        command.add_argument("--output", required=True)
        command.add_argument("--checkpoint", required=True)
        command.add_argument(
            "--device", choices=["cpu", "cuda"], default="cpu"
        )
        command.add_argument("--epochs", type=int, default=1)
        command.add_argument("--batch-size", type=int, default=4)
        command.add_argument(
            "--max-length",
            type=int,
            default=128 if name == "pretrain" else 512,
        )
        command.add_argument(
            "--lr", type=float, default=1e-3 if name == "pretrain" else 5e-5
        )
        command.add_argument(
            "--weight-decay",
            type=float,
            default=1e-4 if name == "pretrain" else 0.01,
        )
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--threads", type=int, default=4)
        command.add_argument("--train-limit", type=int)
        command.add_argument("--dev-limit", type=int)
        command.add_argument("--test-limit", type=int)
        command.add_argument("--train-fraction", type=float, default=1.0)
        command.add_argument("--eval-fraction", type=float, default=1.0)
        command.add_argument("--accumulation-steps", type=int, default=1)
        command.add_argument("--mixed-precision", action="store_true")
        command.add_argument("--tiny", action="store_true")
    command = commands.add_parser("evaluate")
    command.set_defaults(handler=evaluate)
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--model", default="artifacts/gpt2")
    command.add_argument("--data-file", required=True)
    command.add_argument("--language", default="en")
    command.add_argument("--output", required=True)
    command.add_argument("--limit", type=int)
    command.add_argument("--max-length", type=int, default=512)
    command.add_argument("--batch-size", type=int, default=4)
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    command = commands.add_parser("download")
    command.set_defaults(handler=download)
    command.add_argument("--languages", nargs="+", default=["en"])
    command.add_argument("--output", default="data/xnli")
    args = parser.parse_args()
    if hasattr(args, "epochs") and args.epochs < 1:
        parser.error("epochs must be positive")
    args.handler(args)


if __name__ == "__main__":
    main()
