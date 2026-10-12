"""Explicit datasets, label mappings and padding for the recovered tasks."""

import csv
import json
import random
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

LABELS = ("entailment", "contradiction", "neutral")
LANGUAGES = (
    "en",
    "ar",
    "bg",
    "de",
    "el",
    "es",
    "fr",
    "hi",
    "ru",
    "sw",
    "th",
    "tr",
    "ur",
    "vi",
    "zh",
)


def read_records(path, language=None, limit=None, seed=42, fraction=1.0):
    """Read normalized JSONL or original XNLI TSV with named labels."""
    if not 0 < fraction <= 1 or (limit is not None and limit <= 0):
        raise ValueError("fraction must be in (0, 1]; limit must be positive")
    path = Path(path)
    with path.open(encoding="utf-8-sig") as stream:
        if path.suffix == ".jsonl":
            source = [json.loads(line) for line in stream if line.strip()]
        else:
            source = list(csv.DictReader(stream, delimiter="\t"))
    records = []
    for index, row in enumerate(source):
        if language and row.get("language", language) != language:
            continue
        label = row.get("label", row.get("gold_label"))
        if label == "contradictory":
            label = "contradiction"
        if label not in LABELS:
            raise ValueError(
                f"row {index}: expected named NLI label, got {label!r}"
            )
        premise = row.get("premise", row.get("sentence1"))
        hypothesis = row.get(
            "hypothesis", row.get("hypo", row.get("sentence2"))
        )
        if not premise or not hypothesis:
            raise ValueError(f"row {index}: empty premise or hypothesis")
        records.append(
            dict(
                id=row.get("id", f"{path.name}/{index}"),
                premise=premise,
                hypothesis=hypothesis,
                label=label,
                language=row.get("language", language or "en"),
            )
        )
    if not records:
        raise ValueError(
            f"no examples for {language or 'requested split'}: {path}"
        )
    count = max(1, int(len(records) * fraction))
    if limit is not None:
        count = min(count, limit)
    if count < len(records):
        records = random.Random(seed).sample(records, count)
    return records


class NLIDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=512, training=False):
        if not records:
            raise ValueError("dataset must contain examples")
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.training = training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        tokenizer = self.tokenizer
        label = row["label"].replace("contradictory", "contradiction")
        target = str(LABELS.index(label))
        premise, hypothesis = row["premise"], row["hypothesis"]
        if row.get("language") == "zh":
            premise, hypothesis = (
                premise.replace(" ", ""),
                hypothesis.replace(" ", ""),
            )
        encode = tokenizer.encode
        label_ids = encode(target, add_special_tokens=False)
        prefix = encode(
            f"Premise: {premise}\nHypothesis: {hypothesis}\nLabel:",
            add_special_tokens=False,
        )
        budget = self.max_length - len(label_ids)
        if len(prefix) > budget:
            # preserve the hypothesis and label delimiter during truncation.
            start = encode("Premise:", add_special_tokens=False)
            middle = encode("\nHypothesis:", add_special_tokens=False)
            end = encode("\nLabel:", add_special_tokens=False)
            remaining = budget - len(start + middle + end)
            if remaining < 2:
                raise ValueError("max_length cannot fit prompt and target")
            first = encode(" " + premise, add_special_tokens=False)
            second = encode(" " + hypothesis, add_special_tokens=False)
            left = min(len(first), remaining // 2)
            right = min(len(second), remaining - left)
            left = min(len(first), remaining - right)
            prefix = start + first[:left] + middle + second[:right] + end
        ids = prefix + label_ids if self.training else prefix
        output = dict(
            input_ids=torch.tensor(ids, dtype=torch.long),
            attention_mask=torch.ones(len(ids), dtype=torch.long),
            label=target,
            record=row,
        )
        if self.training:
            output["labels"] = torch.tensor(
                [-100] * len(prefix) + label_ids, dtype=torch.long
            )
        return output


class TextDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=128, limit=None):
        self.texts = [
            line
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ][:limit]
        self.tokenizer = tokenizer
        self.max_length = max_length
        if max_length < 2 or not self.texts:
            raise ValueError(
                "pretraining needs nonempty text and max_length >= 2"
            )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        ids = self.tokenizer.encode(
            self.texts[index], add_special_tokens=False
        )
        ids = ids[: self.max_length - 1] + [self.tokenizer.eos_token_id]
        ids = torch.tensor(ids, dtype=torch.long)
        return dict(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            labels=ids.clone(),
        )


def collate_batch(examples, pad_token_id=0):
    if not examples:
        raise ValueError("batch must not be empty")
    batch = {
        key: pad_sequence(
            [example[key] for example in examples],
            batch_first=True,
            padding_value=padding,
        )
        for key, padding in (
            ("input_ids", pad_token_id),
            ("attention_mask", 0),
        )
    }
    if "labels" in examples[0]:
        batch["labels"] = pad_sequence(
            [example["labels"] for example in examples],
            batch_first=True,
            padding_value=-100,
        )
    if "label" in examples[0]:
        batch["label_strs"] = [example["label"] for example in examples]
        batch["records"] = [example["record"] for example in examples]
    return batch
