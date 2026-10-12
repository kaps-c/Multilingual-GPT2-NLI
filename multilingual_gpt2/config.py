from dataclasses import dataclass


@dataclass
class GPT2Config:
    """Only architecture settings used by the custom model."""

    vocab_size: int = 50257
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 1024
    hidden_dropout_prob: float = 0.1
    embedding_dropout_prob: float | None = None
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    hidden_act: str = "gelu_new"

    def __post_init__(self):
        dimensions = (
            self.vocab_size,
            self.hidden_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.intermediate_size,
            self.max_position_embeddings,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("model dimensions must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError(
                "hidden size must be divisible by attention heads"
            )
        if self.hidden_act not in {"gelu", "gelu_new"}:
            raise ValueError("supported activations: gelu, gelu_new")
