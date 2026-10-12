"""Tone and pragmatic inference recovered from PDF pages 102–141."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .cli import load_tokenizer, write_json
from .data import collate_batch
from .evaluation import generate_gpt2
from .model import GPT2Model
from .sentiment import classification_metrics, fit_prompt, parse_label
from .training import load_checkpoint

TONES = ("implicit_care", "disappointment", "comparison", "guilt")
INTENTS = {
    "implicit_care": (
        "The parent is expressing care.",
        "The parent is worried about the child.",
    ),
    "disappointment": (
        "The parent is expressing disappointment.",
        "The parent feels the child did not meet expectations.",
    ),
    "comparison": (
        "The parent is comparing the child to other people.",
        "The parent is using other children as a benchmark.",
    ),
    "guilt": (
        "The parent is trying to make the child feel guilty.",
        "The parent is emphasising their own sacrifices.",
    ),
}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def derive_pragmatic_nli(records):
    pairs = []
    for row in records:
        if row["tone"] not in TONES:
            raise ValueError(f"unknown tone: {row['tone']}")
        for intent, hypotheses in INTENTS.items():
            for index, hypothesis in enumerate(hypotheses):
                pairs.append(
                    {
                        "id": f"{row['id']}/{intent}/{index}",
                        "source_id": row["id"],
                        "premise_id": row["id"],
                        "premise": row["text"],
                        "hypothesis": hypothesis,
                        "intent": "care"
                        if intent == "implicit_care"
                        else intent,
                        "tone": row["tone"],
                        "label": "entailment"
                        if intent == row["tone"]
                        else "contradiction",
                        "language": row.get("lang", "en"),
                    }
                )
    return pairs


def grouped_split(records, seed=42, dev_fraction=0.2):
    """Keep every hypothesis and translation of a source in one split."""
    from sklearn.model_selection import train_test_split

    groups = [
        row.get(
            "source_id",
            row.get("src_id", row.get("premise_id", row.get("id"))),
        )
        for row in records
    ]
    if any(group is None for group in groups):
        raise ValueError("every record needs a source identifier")
    unique = list(dict.fromkeys(groups))
    tones = {
        group: row.get("tone")
        for group, row in zip(groups, records, strict=True)
    }
    strata = [tones[group] for group in unique]
    train, dev = train_test_split(
        unique,
        test_size=dev_fraction,
        random_state=seed,
        stratify=strata if all(strata) else None,
    )
    train, dev = set(train), set(dev)
    return (
        [r for r, g in zip(records, groups, strict=True) if g in train],
        [r for r, g in zip(records, groups, strict=True) if g in dev],
    )


@torch.no_grad()
def text_embeddings(texts, model, tokenizer, device="cpu", max_length=256):
    model.eval()
    vectors = []
    for start in range(0, len(texts), 4):
        examples = []
        for text in texts[start : start + 4]:
            ids = tokenizer.encode(text, add_special_tokens=False)[:max_length]
            if not ids:
                raise ValueError("cannot embed empty text")
            ids = torch.tensor(ids, dtype=torch.long)
            examples.append(
                {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            )
        batch = collate_batch(examples, tokenizer.pad_token_id)
        hidden = model(
            batch["input_ids"].to(device), batch["attention_mask"].to(device)
        )["last_token"]
        vectors.append(hidden.float().cpu().numpy())
    if not vectors:
        raise ValueError("no texts to embed")
    return np.concatenate(vectors)


def fit_probe(train_vectors, train_labels, dev_vectors, dev_labels, seed=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix

    classifier = LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs", random_state=seed
    )
    classifier.fit(train_vectors, train_labels)
    predictions = classifier.predict(dev_vectors).tolist()
    return classifier, {
        **classification_metrics(dev_labels, predictions, classifier.classes_),
        "predictions": predictions,
        "label_order": classifier.classes_.tolist(),
        "confusion_matrix": confusion_matrix(
            dev_labels, predictions, labels=classifier.classes_
        ).tolist(),
    }


def tone_prompt(text, support=()):
    examples = "\n\n".join(
        f"Parent message: {row['text']}\nTone: {row['tone']}"
        for row in support
    )
    instruction = (
        "Classify this Singaporean parent message by pragmatic tone.\n"
        f"Possible tones: {', '.join(TONES)}.\n"
        "Answer with exactly one tone label.\n"
    )
    return instruction + examples + f"\nParent message: {text}\nTone:"


def probe_experiment(records, model, tokenizer, task="tone", device="cpu"):
    train, dev = grouped_split(records)

    def texts(rows):
        if task == "tone":
            return [row["text"] for row in rows]
        return [
            f"Premise: {row['premise']}\n"
            f"Hypothesis: {row['hypothesis']}\nLabel:"
            for row in rows
        ]

    key = "tone" if task == "tone" else "label"
    train_vectors = text_embeddings(texts(train), model, tokenizer, device)
    dev_vectors = text_embeddings(texts(dev), model, tokenizer, device)
    classifier, report = fit_probe(
        train_vectors,
        [row[key] for row in train],
        dev_vectors,
        [row[key] for row in dev],
    )
    report.update(
        train_count=len(train),
        dev_count=len(dev),
        train_ids=[row["id"] for row in train],
        dev_ids=[row["id"] for row in dev],
    )
    return classifier, report, dev_vectors, dev


def main():
    import joblib

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task",
        choices=["derive", "tone", "pragmatic", "style", "cross-lingual"],
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--model", default="artifacts/gpt2")
    parser.add_argument("--probe")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    records = read_jsonl(args.data)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.task == "derive":
        pairs = derive_pragmatic_nli(records)
        with (output / "pragmatic.jsonl").open(
            "w", encoding="utf-8"
        ) as stream:
            for row in pairs:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        return
    if not args.checkpoint:
        parser.error("--checkpoint is required for model experiments")
    torch.set_num_threads(4)
    model, _ = load_checkpoint(args.checkpoint, args.device)
    tokenizer = load_tokenizer(args.model)
    if args.task in {"tone", "pragmatic"}:
        classifier, report, vectors, dev = probe_experiment(
            records, model, tokenizer, args.task, args.device
        )
        bundle = {
            "classifier": classifier,
            "checkpoint": args.checkpoint,
            "train_ids": report["train_ids"],
            "dev_ids": report["dev_ids"],
        }
        joblib.dump(bundle, output / "probe.pkl")
        reloaded = joblib.load(output / "probe.pkl")
        assert (
            reloaded["classifier"].predict(vectors).tolist()
            == report["predictions"]
        )
        from .analysis import embedding_summary

        labels = [
            row["tone" if args.task == "tone" else "label"] for row in dev
        ]
        write_json(
            output / "embeddings.json", embedding_summary(vectors, labels)
        )
        if args.task == "tone":
            generator = GPT2Model.from_pretrained(args.model).to(args.device)
            # support examples come only from the training partition.
            training, _ = grouped_split(records)
            support = [
                next(row for row in training if row["tone"] == tone)
                for tone in TONES
            ]
            for mode, examples in (("zero", []), ("few", support)):
                predictions = []
                for row in dev:
                    template = (
                        tone_prompt("{text}", examples)
                        .replace("{", "{{")
                        .replace("}", "}}")
                        .replace("{{text}}", "{text}")
                    )
                    prompt = fit_prompt(template, row["text"], tokenizer, 512)
                    continuation = generate_gpt2(
                        generator, tokenizer, prompt, 12, args.device
                    )
                    predictions.append(parse_label(continuation, TONES))
                report[mode] = classification_metrics(
                    [row["tone"] for row in dev], predictions, TONES
                )
        write_json(output / "results.json", report)
        return
    if not args.probe:
        parser.error(
            "--probe is required for style and cross-lingual experiments"
        )
    bundle = joblib.load(args.probe)
    probe_model, _ = load_checkpoint(bundle["checkpoint"], args.device)
    generated = []
    if args.task == "style":
        generator = GPT2Model.from_pretrained(args.model).to(args.device)
        for row in records:
            neutral = row.get(
                "neutral_context", row.get("neutral", row.get("text"))
            )
            if not neutral:
                raise ValueError("style record needs neutral_context")
            for tone in TONES:
                prompt = (
                    "Rewrite this Singaporean parent message in Singlish.\n"
                    f"Tone: {tone}\nNeutral message: {neutral}\nRewrite:"
                )
                text = generate_gpt2(
                    generator, tokenizer, prompt, 40, args.device
                )
                generated.append(
                    {
                        "source_id": row.get("id", row.get("situation_id")),
                        "text": text,
                        "tone": tone,
                    }
                )
    else:
        generated = records
    vectors = text_embeddings(
        [row["text"] or " " for row in generated],
        probe_model,
        tokenizer,
        args.device,
    )
    predictions = bundle["classifier"].predict(vectors).tolist()
    predictions = [
        prediction if row["text"].strip() else "unknown"
        for row, prediction in zip(generated, predictions, strict=True)
    ]
    report = classification_metrics(
        [row["tone"] for row in generated], predictions, TONES
    )
    report["examples"] = [
        {**row, "prediction": prediction}
        for row, prediction in zip(generated, predictions, strict=True)
    ]
    report["note"] = (
        "probe tone agreement does not measure preservation of meaning"
    )
    if args.task == "cross-lingual":
        train_ids = set(bundle["train_ids"])
        held_out = [
            index
            for index, row in enumerate(generated)
            if row.get("source_id", row.get("src_id")) not in train_ids
            and row.get("source_id", row.get("src_id")) is not None
        ]
        report["held_out_source_count"] = len(held_out)
        report["held_out_source_metrics"] = (
            classification_metrics(
                [generated[index]["tone"] for index in held_out],
                [predictions[index] for index in held_out],
                TONES,
            )
            if held_out
            else None
        )
        report["note"] = (
            "all-row metrics may include translations of probe training "
            "examples; held-out metrics require known, unseen source ids"
        )
    write_json(output / "results.json", report)


if __name__ == "__main__":
    main()
