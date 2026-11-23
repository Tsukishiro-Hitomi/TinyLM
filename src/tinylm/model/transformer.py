"""Transformer language-model components."""

import torch
import torch.nn as nn

from .attention import MultiheadSelfAttention, RoPE
from .layers import Embedding, Linear, RMSNorm


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None):
        super().__init__()

        self.d_model = d_model
        if d_ff is None:
            d_ff = int(d_model * 8 / 3)
            if d_ff % 64 != 0:
                d_ff -= d_ff % 64
        self.d_ff = d_ff

        self.linear1 = Linear(self.d_model, self.d_ff)
        self.linear2 = Linear(self.d_ff, self.d_model)
        self.linear3 = Linear(self.d_model, self.d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear1(x)
        out = silu(out)
        out = out * self.linear3(x)
        return self.linear2(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RoPE,
        attn_implementation: str = "manual",
    ):
        super().__init__()
        self.swiglu = SwiGLU(d_model, d_ff)
        self.attention = MultiheadSelfAttention(d_model, num_heads, attn_implementation=attn_implementation)
        self.rmsnorm1 = RMSNorm(d_model)
        self.rmsnorm2 = RMSNorm(d_model)
        self.rope = rope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.forward_cached(x)
        return out

    def forward_cached(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        token_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_kv = self.attention.forward_cached(
            self.rmsnorm1(x), self.rope, token_positions, past_kv
        )
        y = x + attn_out
        return y + self.swiglu(self.rmsnorm2(y)), new_kv


class TransformerLM(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RoPE,
        vocab_size: int,
        num_layers: int,
        device: torch.device | None = None,
        attn_implementation: str = "manual",
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = Embedding(vocab_size, self.d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, rope, attn_implementation=attn_implementation)
                for _ in range(num_layers)
            ]
        )
        self.rmsnorm = RMSNorm(d_model)
        self.final_linear = Linear(d_model, vocab_size)

        # Submodules are created on the default device; move them so that the
        # ``device`` argument is honored end to end (the original assignment code
        # relied on an external ``model.to(device)`` call instead).
        if device is not None:
            self.to(device)
        self.device = device

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_cached(in_indices)
        return logits

    def forward_cached(
        self,
        in_indices: torch.Tensor,
        past_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass that returns per-layer ``(k, v)`` caches for decoding."""
        # in_indices is 2-D (batch, seq); the sequence axis is the last one.
        length = in_indices.shape[-1]
        positions = torch.arange(start_pos, start_pos + length, device=in_indices.device, dtype=torch.long)

        x = self.embedding(in_indices)
        new_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for index, block in enumerate(self.blocks):
            layer_past = past_kv[index] if past_kv is not None else None
            x, layer_kv = block.forward_cached(x, past_kv=layer_past, token_positions=positions)
            new_kv.append(layer_kv)

        x = self.rmsnorm(x)
        logits = self.final_linear(x)
        return logits, new_kv
