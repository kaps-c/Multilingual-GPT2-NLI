import json
from types import SimpleNamespace

from multilingual_gpt2.synthetic import generate_dataset
from multilingual_gpt2.tone import TONES, read_jsonl


def test_translation_budget_samples_each_tone_and_preserves_source(tmp_path):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="text",
                    text=json.dumps({"text": "translated message"}),
                )
            ]
        )

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    core = [
        {"id": f"{tone}/{i}", "text": f"message {i}", "tone": tone}
        for tone in TONES
        for i in range(5)
    ]
    path = tmp_path / "translations.jsonl"
    generate_dataset(client, "test-model", "translation", path, 4, core)
    rows = read_jsonl(path)
    assert len(calls) == len(rows) == 4
    assert {row["tone"] for row in rows} == set(TONES)
    assert all(row["source_id"] in {r["id"] for r in core} for row in rows)
    assert all(row["source"] == "new_synthetic_generation" for row in rows)
