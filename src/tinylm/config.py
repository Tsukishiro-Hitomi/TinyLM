"""Typed configuration objects and model factory.

The dataclasses here centralize the hyperparameters that were previously
scattered as positional/keyword arguments across the training and generation
scripts, so that a single, serializable config can drive model construction,
training, and inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .model.attention import RoPE
from .model.transformer import TransformerLM


@dataclass
class ModelConfig:
    """Architecture hyperparameters for :class:`TransformerLM`."""

    d_model: int = 512
    num_heads: int = 16
    d_ff: int = 1344
    vocab_size: int = 10_000
    num_layers: int = 4
    rope_theta: float = 10_000.0
    max_seq_len: int = 256
    tie_weights: bool = False
    attn_implementation: str = "sdpa"

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    def build(self, device: torch.device | str | None = None) -> TransformerLM:
        """Instantiate the language model described by this config."""
        rope = RoPE(
            theta=self.rope_theta,
            d_k=self.head_dim,
            max_seq_len=self.max_seq_len,
            device=_as_device(device),
        )
        model = TransformerLM(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            rope=rope,
            vocab_size=self.vocab_size,
            num_layers=self.num_layers,
            device=_as_device(device),
            attn_implementation=self.attn_implementation,
        )
        if self.tie_weights:
            # Share the input embedding and output projection weights.
            model.final_linear.W = model.embedding.embedding
        return model


@dataclass
class TrainConfig:
    """Training-loop hyperparameters."""

    batch_size: int = 32
    context_length: int = 256
    steps: int = 40_000
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_ratio: float = 0.05
    weight_decay: float = 0.1
    save_every: int = 1_000
    evaluate_every: int = 200
    device: str = "auto"
    seed: int = 336


@dataclass
class TokenizerConfig:
    """Tokenizer-training hyperparameters."""

    vocab_size: int = 10_000
    special_tokens: list[str] = field(default_factory=lambda: ["<|endoftext|>"])
    num_chunks: int = 32


def _as_device(device: torch.device | str | None) -> torch.device | None:
    if device is None:
        return None
    return torch.device(device)
