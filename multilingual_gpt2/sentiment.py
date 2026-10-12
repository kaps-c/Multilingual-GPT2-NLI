"""Continuation-only prompt classification for PDF pages 38–55."""

import re

from .evaluation import generate_gpt2

SENTIMENT_LABELS = ("negative", "neutral", "positive")
FEW_SHOT = (
    "Tweet: I love this product, it's amazing!\nSentiment: positive\n\n"
    "Tweet: It's okay, nothing special.\nSentiment: neutral\n\n"
    "Tweet: I hate this, it was a waste of money.\nSentiment: negative\n\n"
)


def normalize_label(value):
    label = str(value)
    if label in {"0", "1", "2"}:
        return SENTIMENT_LABELS[int(label)]
    if label in SENTIMENT_LABELS:
        return label
    raise ValueError(f"unexpected sentiment label: {value!r}")


def parse_label(continuation, labels, strict=True):
    pattern = "|".join(re.escape(label) for label in labels)
    text = continuation.strip().lower()
    expression = rf"^({pattern})(?=\W|$)" if strict else rf"\b({pattern})\b"
    match = re.search(expression, text)
    return match.group(1) if match else "unknown"


def classification_metrics(gold, predictions, labels):
    from sklearn.metrics import f1_score

    if not gold or len(gold) != len(predictions):
        raise ValueError(
            "gold and predictions must have matching nonempty rows"
        )
    if any(label not in labels for label in gold):
        raise ValueError("unexpected gold label")
    return {
        "accuracy": sum(a == b for a, b in zip(gold, predictions, strict=True))
        / len(gold),
        "macro_f1": float(
            f1_score(
                gold,
                predictions,
                labels=list(labels),
                average="macro",
                zero_division=0,
            )
        ),
        "coverage": sum(label in labels for label in predictions) / len(gold),
        "count": len(gold),
    }


def fit_prompt(template, text, tokenizer, max_length):
    """Truncate the input text while retaining the task's output delimiter."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    prompt = template.format(text=text)
    while len(tokenizer.encode(prompt, add_special_tokens=False)) > max_length:
        if not tokens:
            raise ValueError("prompt template exceeds the context budget")
        tokens = tokens[: -max(1, len(tokens) // 8)]
        prompt = template.format(text=tokenizer.decode(tokens))
    return prompt


def evaluate_sentiment(
    records,
    model,
    tokenizer,
    mode="strict",
    device="cpu",
    max_length=128,
):
    if mode not in {"zero", "few", "strict"}:
        raise ValueError("sentiment mode must be zero, few, or strict")
    template = "Tweet: {text}\nSentiment (negative, neutral, positive):"
    if mode == "few":
        template = FEW_SHOT + "Tweet: {text}\nSentiment:"
    elif mode == "strict":
        template = (
            "Classify the sentiment of this tweet.\n"
            "Answer with exactly one word (negative, neutral, positive).\n"
            "Tweet: {text}\nLabel:"
        )
    rows = []
    for row in records:
        prompt = fit_prompt(template, row["text"], tokenizer, max_length)
        continuation = generate_gpt2(model, tokenizer, prompt, 5, device)
        rows.append(
            {
                **row,
                "continuation": continuation,
                "prediction": parse_label(
                    continuation, SENTIMENT_LABELS, strict=mode == "strict"
                ),
            }
        )
    return {
        **classification_metrics(
            [row["label"] for row in rows],
            [row["prediction"] for row in rows],
            SENTIMENT_LABELS,
        ),
        "mode": mode,
        "predictions": rows,
    }


def main():
    import argparse
    import json
    from pathlib import Path

    import torch

    from .cli import load_tokenizer, write_json
    from .training import load_checkpoint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--model", default="artifacts/gpt2")
    parser.add_argument("--data", default="data/sentiment")
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["arabic", "english", "french", "german", "hindi", "spanish"],
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", default="artifacts/sentiment")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.download:
        from datasets import load_dataset

        for language in args.languages:
            path = Path(args.data) / f"{language}.jsonl"
            if path.exists():
                continue
            dataset = load_dataset(
                "mteb/tweet_sentiment_multilingual",
                language,
                split="test",
                cache_dir=str(Path(args.data) / ".cache"),
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as stream:
                for index, row in enumerate(dataset):
                    label = normalize_label(row["label"])
                    stream.write(
                        json.dumps(
                            {
                                "id": f"{language}/{index}",
                                "text": row["text"],
                                "label": label,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    if not args.checkpoint:
        if args.download:
            return
        parser.error("provide --checkpoint or --download")
    torch.set_num_threads(4)
    tokenizer = load_tokenizer(args.model)
    model, _ = load_checkpoint(args.checkpoint, args.device)
    summaries = []
    for language in args.languages:
        path = Path(args.data) / f"{language}.jsonl"
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ][: args.limit]
        records = [
            {**row, "label": normalize_label(row["label"])} for row in records
        ]
        for mode in ("zero", "few", "strict"):
            result = evaluate_sentiment(
                records, model, tokenizer, mode, args.device
            )
            result.update(language=language, checkpoint=args.checkpoint)
            write_json(Path(args.output) / f"{language}-{mode}.json", result)
            summaries.append(
                {
                    key: value
                    for key, value in result.items()
                    if key != "predictions"
                }
            )
            print(f"{language}/{mode}: {result['accuracy']:.3f}", flush=True)
    write_json(Path(args.output) / "summary.json", summaries)


if __name__ == "__main__":
    main()
