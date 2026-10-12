import numpy as np

from multilingual_gpt2.analysis import calibration_metrics


def test_multiclass_calibration_counts_confident_errors():
    probabilities = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    result = calibration_metrics(probabilities, np.array([0, 0]))
    assert result["accuracy"] == 0.5
    assert result["brier"] == 1.0
    assert result["ece"] == 0.5
    assert sum(row["count"] for row in result["bins"]) == 2


def test_attention_mass_uses_delimiters_in_the_actual_tokenized_prompt():
    from multilingual_gpt2.analysis import prompt_attention_mass

    class Tokenizer:
        def encode(self, text, **kwargs):
            return [ord(char) for char in text]

    tokenizer = Tokenizer()
    ids = tokenizer.encode("Premise: a\nHypothesis: b\nLabel:")
    weights = np.zeros((2, len(ids), len(ids)))
    weights[:, :, 0] = 1
    result = prompt_attention_mass(weights, ids, tokenizer)
    assert result["hypothesis_to_premise"] == 1
    assert result["hypothesis_to_hypothesis"] == 0
