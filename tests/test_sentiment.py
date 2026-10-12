from multilingual_gpt2.sentiment import classification_metrics, parse_label


def test_dataset_labels_accept_numeric_strings_without_hiding_invalid_data():
    import pytest

    from multilingual_gpt2.sentiment import normalize_label

    assert [normalize_label(x) for x in ("0", "1", "2")] == [
        "negative",
        "neutral",
        "positive",
    ]
    assert normalize_label(0) == "negative"
    assert normalize_label("positive") == "positive"
    with pytest.raises(ValueError, match="sentiment label"):
        normalize_label("unknown")


def test_unknown_generations_are_wrong_without_reading_prompt_labels():
    labels = ("negative", "neutral", "positive")
    assert parse_label("positive\nTweet: next", labels) == "positive"
    assert parse_label("I cannot tell", labels) == "unknown"
    metrics = classification_metrics(
        ["positive", "negative"], ["positive", "unknown"], labels
    )
    assert metrics["accuracy"] == 0.5
    assert metrics["coverage"] == 0.5
    assert metrics["count"] == 2
