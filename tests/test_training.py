import torch

from multilingual_gpt2.config import GPT2Config
from multilingual_gpt2.model import GPT2Model
from multilingual_gpt2.training import (
    load_checkpoint,
    save_checkpoint,
    train_epoch,
)


def test_training_updates_weights_and_checkpoint_reloads(tmp_path):
    torch.manual_seed(2)
    model = GPT2Model(
        GPT2Config(
            vocab_size=31,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=8,
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0,
        )
    )
    ids = torch.tensor([[2, 4, 6]])
    batch = dict(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        labels=torch.tensor([[-100, -100, 6]]),
    )
    before = model.word_embedding.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    losses = train_epoch(model, [batch], optimizer, "cpu")
    assert len(losses) == 1 and losses[0] > 0
    assert not torch.equal(before, model.word_embedding.weight)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(model, path, {"dev_accuracy": 0.0})
    restored, metadata = load_checkpoint(path)
    model.eval()
    restored.eval()
    torch.testing.assert_close(
        model(ids)["last_hidden_state"], restored(ids)["last_hidden_state"]
    )
    assert metadata["dev_accuracy"] == 0.0
