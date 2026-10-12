import torch
from transformers import GPT2Config as ReferenceConfig
from transformers import GPT2Model as ReferenceModel

from multilingual_gpt2.model import GPT2Model


def test_imported_weights_match_reference_and_causal_mask():
    torch.manual_seed(7)
    reference = ReferenceModel(
        ReferenceConfig(
            vocab_size=31,
            n_positions=16,
            n_embd=16,
            n_layer=2,
            n_head=4,
            n_inner=64,
            attn_pdrop=0,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_implementation="eager",
        )
    ).eval()
    model = GPT2Model.from_reference(reference).eval()
    ids = torch.tensor([[1, 4, 7, 2], [1, 3, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    expected = reference(ids, attention_mask=mask).last_hidden_state
    actual = model(ids, mask)["last_hidden_state"]
    torch.testing.assert_close(actual[mask.bool()], expected[mask.bool()])
    changed = ids.clone()
    changed[0, -1] = 9
    torch.testing.assert_close(
        actual[0, :-1], model(changed, mask)["last_hidden_state"][0, :-1]
    )
    assert model.config.intermediate_size == 64


def test_checkpoint_preserves_distinct_embedding_dropout(tmp_path):
    from multilingual_gpt2.training import load_checkpoint, save_checkpoint

    reference = ReferenceModel(
        ReferenceConfig(
            vocab_size=31,
            n_positions=8,
            n_embd=8,
            n_layer=1,
            n_head=2,
            resid_pdrop=0,
            embd_pdrop=0.5,
            attn_pdrop=0,
        )
    )
    model = GPT2Model.from_reference(reference).train()
    path = tmp_path / "dropout.pt"
    save_checkpoint(model, path)
    restored, _ = load_checkpoint(path)
    restored.train()
    ids = torch.tensor([[1, 2, 3, 4]])
    torch.manual_seed(13)
    before = model(ids)["last_hidden_state"]
    torch.manual_seed(13)
    after = restored(ids)["last_hidden_state"]
    torch.testing.assert_close(before, after)
