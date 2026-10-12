"""Explicit, bounded synthetic-data generation from PDF pages 102–115."""

import argparse
import hashlib
import json
import random
from itertools import zip_longest
from pathlib import Path

from .tone import TONES, read_jsonl

SYSTEM = (
    "Generate fictional examples for a linguistics research dataset about "
    "Singaporean parent communication. Return only the requested JSON."
)
DEFINITIONS = (
    "implicit_care: indirect affection or concern; disappointment: direct "
    "negative evaluation without comparison; comparison: comparing with "
    "siblings or other children; guilt: emphasising sacrifice or duty."
)


def parse_response(response):
    text = "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def generate_dataset(client, model, kind, output, max_calls, core_records=()):
    """Persist each response so interrupted generation remains inspectable."""
    if max_calls < 1:
        raise ValueError("max_calls must be positive")
    output = Path(output)
    if output.exists():
        raise FileExistsError(
            "choose a new output file to preserve existing data"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if kind != "core" and not core_records:
        raise ValueError(
            "style and translation generation require core records"
        )
    seen = set()
    if kind == "core":
        requests = range(max_calls)
    else:
        buckets = [
            [row for row in core_records if row["tone"] == tone]
            for tone in TONES
        ]
        rng = random.Random(42)
        for bucket in buckets:
            rng.shuffle(bucket)
        requests = [
            row
            for group in zip_longest(*buckets)
            for row in group
            if row is not None
        ][:max_calls]
    with output.open("w", encoding="utf-8") as stream:
        for item in requests:
            if kind == "core":
                prompt = (
                    f"{DEFINITIONS}\nGenerate 8 Singlish messages per tone. "
                    "Use one sentence, at most 25 words, without vulgarities. "
                    "Vary school, chores, money, health and relationships. "
                    "Return a JSON array of objects with text and tone keys."
                )
            elif kind == "style":
                prompt = (
                    f"{DEFINITIONS}\nOriginal message: {item['text']}\n"
                    "Return a JSON object with neutral_context (the intention "
                    "without tone) and rewrites (one short Singlish sentence "
                    "for each of implicit_care, disappointment, comparison, "
                    "guilt). Preserve the underlying situation."
                )
            else:
                prompt = (
                    f"Translate into natural Mandarin used by Singaporean "
                    f"Chinese parents. Preserve the tone {item['tone']} and "
                    f"meaning: {item['text']}\nReturn JSON with a text key."
                )
            response = client.messages.create(
                model=model,
                max_tokens={"core": 2048, "style": 512, "translation": 256}[
                    kind
                ],
                temperature={"core": 0.75, "style": 0.7, "translation": 0.6}[
                    kind
                ],
                system=SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            generated = parse_response(response)
            rows = generated if kind == "core" else [generated]
            if not isinstance(rows, list):
                raise ValueError("core generation must return an array")
            for row in rows:
                if kind == "style":
                    if not row.get("neutral_context") or set(
                        row.get("rewrites", {})
                    ) != set(TONES):
                        raise ValueError(
                            "style output needs neutral context and all tones"
                        )
                    row.update(source_id=item["id"], base_text=item["text"])
                else:
                    if kind == "translation":
                        row.update(tone=item["tone"], source_id=item["id"])
                    if (
                        row.get("tone") not in TONES
                        or not row.get("text", "").strip()
                    ):
                        raise ValueError(
                            "generated row has missing text or invalid tone"
                        )
                signature = hashlib.sha256(
                    json.dumps(row, sort_keys=True).encode()
                ).hexdigest()
                if signature in seen:
                    continue
                seen.add(signature)
                row.update(
                    id=signature[:24],
                    source="new_synthetic_generation",
                    model=model,
                    lang="zh" if kind == "translation" else "en",
                )
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["core", "style", "translation"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-calls", type=int, required=True)
    parser.add_argument("--core")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.kind != "core" and not args.core:
        parser.error("--core is required for style and translation")
    import anthropic

    client = anthropic.Anthropic()
    generate_dataset(
        client,
        args.model,
        args.kind,
        args.output,
        args.max_calls,
        read_jsonl(args.core) if args.core else (),
    )


if __name__ == "__main__":
    main()
