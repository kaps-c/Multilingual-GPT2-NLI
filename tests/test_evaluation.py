import torch

from multilingual_gpt2.evaluation import evaluate_nli


class Tokenizer:
    def encode(self, text, **kwargs):
        return [int(text)]

    def decode(self, token_ids, **kwargs):
        return str(token_ids[0])


class Model(torch.nn.Module):
    def forward(self, input_ids, attention_mask):
        return {"last_token": torch.zeros(len(input_ids), 1)}

    def hidden_state_to_token(self, hidden):
        return torch.tensor([[9.0, 0, 0, 0], [0, 0, 0, 9.0]])


def test_evaluation_keeps_invalid_predictions_in_denominator():
    batch = {
        "input_ids": torch.ones(2, 2, dtype=torch.long),
        "attention_mask": torch.ones(2, 2, dtype=torch.long),
        "label_strs": ["0", "1"],
        "records": [{"id": "a"}, {"id": "b"}],
    }
    result = evaluate_nli(Model(), Tokenizer(), [batch], "cpu")
    assert result["accuracy"] == 0.5
    assert result["coverage"] == 0.5
    assert result["count"] == 2
    assert len(result["predictions"]) == 2
