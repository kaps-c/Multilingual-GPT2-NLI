from multilingual_gpt2.tone import derive_pragmatic_nli, grouped_split


def test_pragmatic_pairs_from_same_premise_never_cross_splits():
    records = [
        {"id": str(index), "text": f"message {index}", "tone": tone}
        for index, tone in enumerate(["implicit_care", "guilt"] * 8)
    ]
    pairs = derive_pragmatic_nli(records)
    train, dev = grouped_split(pairs, seed=42)
    assert {row["source_id"] for row in train}.isdisjoint(
        row["source_id"] for row in dev
    )
    assert len(train) + len(dev) == len(pairs)
    assert {row["label"] for row in pairs} == {"entailment", "contradiction"}


def test_translations_keep_source_groups_and_all_tones_in_training():
    from multilingual_gpt2.tone import TONES

    records = [
        {
            "id": f"{tone}/{index}/{lang}",
            "src_id": f"{tone}/{index}",
            "tone": tone,
        }
        for tone in TONES
        for index in range(5)
        for lang in ("en", "zh")
    ]
    train, dev = grouped_split(records)
    assert {r["src_id"] for r in train}.isdisjoint(r["src_id"] for r in dev)
    assert {r["tone"] for r in train} == set(TONES)
    assert {r["tone"] for r in dev} == set(TONES)
