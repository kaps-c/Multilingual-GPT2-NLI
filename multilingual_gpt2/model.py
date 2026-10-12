"""Custom GPT-2, recovered from notebook cells 5–8."""

import torch
from torch import nn
from torch.nn import functional as F

from .config import GPT2Config


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states, attention_mask, output_attentions=False):
        batch, length, width = hidden_states.shape

        def project(layer):
            return (
                layer(hidden_states)
                .view(batch, length, self.num_attention_heads, -1)
                .transpose(1, 2)
            )

        query, key, value = map(project, (self.query, self.key, self.value))
        allowed = (
            torch.ones(
                length, length, dtype=torch.bool, device=hidden_states.device
            ).tril()[None, None]
            & attention_mask[:, None, None, :].bool()
        )
        weights = None
        if output_attentions:
            scores = query @ key.transpose(-1, -2) / query.shape[-1] ** 0.5
            scores = scores.masked_fill(
                ~allowed, torch.finfo(scores.dtype).min
            )
            weights = scores.softmax(-1)
            context = self.dropout(weights) @ value
        else:
            context = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=allowed,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        return context.transpose(1, 2).reshape(batch, length, width), weights


class GPT2Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = CausalSelfAttention(config)
        self.attention_dense = nn.Linear(
            config.hidden_size, config.hidden_size
        )
        self.attention_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.interm_dense = nn.Linear(
            config.hidden_size, config.intermediate_size
        )
        self.out_dense = nn.Linear(
            config.intermediate_size, config.hidden_size
        )
        self.out_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.approximate = (
            "tanh" if config.hidden_act == "gelu_new" else "none"
        )

    def forward(self, hidden_states, attention_mask, output_attentions=False):
        context, weights = self.self_attention(
            self.attention_layer_norm(hidden_states),
            attention_mask,
            output_attentions,
        )
        hidden_states = hidden_states + self.attention_dropout(
            self.attention_dense(context)
        )
        intermediate = F.gelu(
            self.interm_dense(self.out_layer_norm(hidden_states)),
            approximate=self.approximate,
        )
        return hidden_states + self.out_dropout(
            self.out_dense(intermediate)
        ), weights


class GPT2Model(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or GPT2Config()
        config = self.config
        # gpt-2 has no dedicated padding embedding; masks identify padding.
        self.word_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size
        )
        self.pos_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.gpt_layers = nn.ModuleList(
            GPT2Layer(config) for _ in range(config.num_hidden_layers)
        )
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask=None, output_attentions=False):
        if input_ids.ndim != 2 or not input_ids.shape[1]:
            raise ValueError("input_ids must be a nonempty batch of sequences")
        if input_ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError("input exceeds model context length")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention mask must match input shape")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = self.embed_dropout(
            self.word_embedding(input_ids) + self.pos_embedding(positions)
        )
        attentions = []
        for layer in self.gpt_layers:
            hidden, weights = layer(hidden, attention_mask, output_attentions)
            if output_attentions:
                attentions.append(weights)
        hidden = self.final_layer_norm(hidden)
        # last real position works for any padding layout.
        indices = (
            positions.expand_as(input_ids)
            .masked_fill(~attention_mask.bool(), -1)
            .max(dim=1)
            .values
        )
        if (indices < 0).any():
            raise ValueError(
                "each sequence must contain at least one real token"
            )
        output = {
            "last_hidden_state": hidden,
            "last_token": hidden[
                torch.arange(len(hidden), device=hidden.device), indices
            ],
        }
        if output_attentions:
            output["attentions"] = tuple(attentions)
        return output

    def hidden_state_to_token(self, hidden_state):
        """Project hidden states through the tied token embedding weights."""
        return F.linear(hidden_state, self.word_embedding.weight)

    @classmethod
    def from_pretrained(cls, name="gpt2", **kwargs):
        from transformers import GPT2Model as ReferenceModel

        return cls.from_reference(
            ReferenceModel.from_pretrained(name, **kwargs)
        )

    @classmethod
    def from_reference(cls, reference):
        """Copy official weights without changing parameter storage shapes."""
        source = reference.config
        model = cls(
            GPT2Config(
                vocab_size=source.vocab_size,
                hidden_size=source.n_embd,
                num_hidden_layers=source.n_layer,
                num_attention_heads=source.n_head,
                intermediate_size=source.n_inner or source.n_embd * 4,
                max_position_embeddings=source.n_positions,
                hidden_dropout_prob=source.resid_pdrop,
                attention_probs_dropout_prob=source.attn_pdrop,
                layer_norm_eps=source.layer_norm_epsilon,
                initializer_range=source.initializer_range,
                hidden_act=source.activation_function,
            )
        )
        model.embed_dropout.p = source.embd_pdrop
        with torch.no_grad():
            model.word_embedding.weight.copy_(reference.wte.weight)
            model.pos_embedding.weight.copy_(reference.wpe.weight)
            for target, origin in zip(
                model.gpt_layers, reference.h, strict=True
            ):
                for index, name in enumerate(("query", "key", "value")):
                    linear = getattr(target.self_attention, name)
                    begin, end = (
                        index * source.n_embd,
                        (index + 1) * source.n_embd,
                    )
                    linear.weight.copy_(
                        origin.attn.c_attn.weight[:, begin:end].T
                    )
                    linear.bias.copy_(origin.attn.c_attn.bias[begin:end])
                for linear, conv in (
                    (target.attention_dense, origin.attn.c_proj),
                    (target.interm_dense, origin.mlp.c_fc),
                    (target.out_dense, origin.mlp.c_proj),
                ):
                    linear.weight.copy_(conv.weight.T)
                    linear.bias.copy_(conv.bias)
                target.attention_layer_norm.load_state_dict(
                    origin.ln_1.state_dict()
                )
                target.out_layer_norm.load_state_dict(origin.ln_2.state_dict())
            model.final_layer_norm.load_state_dict(reference.ln_f.state_dict())
        return model.eval()
