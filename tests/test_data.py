import torch

from multilingual_gpt2.data import NLIDataset, collate_batch


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, **kwargs):
        return [2 + ord(char) % 25 for char in text]


def test_long_prompts_keep_label_and_padding_is_not_supervised():
    tokenizer = TinyTokenizer()
    rows = [
        {
            "premise": "x" * 300,
            "hypothesis": "y" * 300,
            "label": "contradiction",
        },
        {"premise": "a", "hypothesis": "b", "label": "neutral"},
    ]
    dataset = NLIDataset(rows, tokenizer, max_length=48, training=True)
    batch = collate_batch([dataset[0], dataset[1]], tokenizer.pad_token_id)
    assert batch["input_ids"].shape[1] <= 48
    for index, target in enumerate(("1", "2")):
        labels = batch["labels"][index]
        assert labels[labels != -100].tolist() == tokenizer.encode(target)
        assert (labels[batch["attention_mask"][index] == 0] == -100).all()
    evaluation = NLIDataset(rows, tokenizer, max_length=48, training=False)
    assert "labels" not in evaluation[0]
    assert torch.equal(
        dataset[0]["input_ids"][:-1], evaluation[0]["input_ids"]
    )
