"""Helpers to reconstruct a model from a legacy checkpoint."""

from __future__ import annotations

import torch

from tinylm import ModelConfig


def infer_model_config(state_dict: dict, num_heads: int = 16, context: int = 256) -> ModelConfig:
    """Derive a :class:`ModelConfig` from the shapes in a state dict."""
    embedding = state_dict["embedding.embedding"]
    vocab_size, d_model = embedding.shape
    num_layers = len({key.split(".")[1] for key in state_dict if key.startswith("blocks.")})
    d_ff = state_dict["blocks.0.swiglu.linear1.W"].shape[0]
    return ModelConfig(
        d_model=int(d_model),
        num_heads=num_heads,
        d_ff=int(d_ff),
        vocab_size=int(vocab_size),
        num_layers=num_layers,
        max_seq_len=context,
        attn_implementation="sdpa",
    )


def load_legacy_state_dict(path, device: torch.device | str = "cpu") -> dict:
    """Load ``(iteration, state_dict, optimizer_state)`` tuples or plain state dicts."""
    payload = torch.load(path, map_location=device, weights_only=False)
    if isinstance(payload, tuple) and len(payload) >= 2:
        return payload[1]
    return payload
