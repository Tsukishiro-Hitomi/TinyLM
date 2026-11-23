"""Attention primitives, rotary position embeddings, and KV-cache support."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Linear


class RoPE(nn.Module):
    """Rotary position embedding, applied to queries and keys."""

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        assert d_k % 2 == 0
        self.d_k = d_k

        # (d_k // 2, )
        k = torch.arange(d_k // 2, device=device)
        # (L, )
        positions = torch.arange(max_seq_len, device=device)
        # (d_k // 2, )
        inv_freq = 1.0 / (theta ** ((2 * k) / d_k))
        # (L, d_k // 2)
        theta_ik = positions[:, None] * inv_freq[None, :]

        self.register_buffer("sin_ik", torch.sin(theta_ik), persistent=False)
        self.register_buffer("cos_ik", torch.cos(theta_ik), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., L, d_k)
        # token_positions: (..., L)
        assert x.shape[-1] == self.d_k
        assert x.shape[-2] == token_positions.shape[-1]

        # (..., L, d_k // 2)
        sin = self.sin_ik[token_positions].to(dtype=x.dtype)
        cos = self.cos_ik[token_positions].to(dtype=x.dtype)
        # (..., L, d_k // 2, 1)
        sin = sin.unsqueeze(-1)
        cos = cos.unsqueeze(-1)

        x_shape = x.shape
        # (..., L, d_k // 2, 2)
        x = x.reshape(*x_shape[:-1], self.d_k // 2, 2)
        first = x[..., 0]
        second = x[..., 1]

        rotated_first = first * cos[..., 0] - second * sin[..., 0]
        rotated_second = first * sin[..., 0] + second * cos[..., 0]
        rotated = torch.stack((rotated_first, rotated_second), dim=-1)

        return rotated.reshape(x_shape)


def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    logits = x.clone()
    logits -= torch.max(logits, dim=i, keepdim=True).values
    exp_logits = torch.exp(logits)
    return exp_logits / torch.sum(exp_logits, dim=i, keepdim=True)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    # Q: (..., querys, d_k)
    # K: (..., keys, d_k)
    # V: (..., keys, d_v)
    # mask: (..., querys, keys)
    d_k = Q.shape[-1]
    scores = (Q @ K.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = softmax(scores, -1)
    return attn @ V


class MultiheadSelfAttention(nn.Module):
    """Causal multi-head self-attention with optional KV cache and SDPA.

    ``forward`` keeps the original signature and returns only the output, so the
    assignment-compatibility tests are unaffected. ``forward_cached`` additionally
    accepts and returns a ``(k, v)`` cache for incremental decoding.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        use_causal_mask: bool = True,
        attn_implementation: str = "manual",
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_causal_mask = use_causal_mask
        self.attn_implementation = attn_implementation

        self.q = Linear(d_model, d_model, device, dtype)
        self.k = Linear(d_model, d_model, device, dtype)
        self.v = Linear(d_model, d_model, device, dtype)
        self.o = Linear(d_model, d_model, device, dtype)

    def forward(
        self,
        features: torch.Tensor,
        rope: RoPE | None,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out, _ = self.forward_cached(features, rope, token_positions)
        return out

    def forward_cached(
        self,
        features: torch.Tensor,
        rope: RoPE | None,
        token_positions: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # features: (..., L_new, D)
        sequence_length = features.shape[-2]

        # (..., L, D) -> (..., H, L, D // H)
        q = self._split_heads(self.q(features)).transpose(-3, -2)
        k = self._split_heads(self.k(features)).transpose(-3, -2)
        v = self._split_heads(self.v(features)).transpose(-3, -2)

        if rope is not None:
            if token_positions is None:
                token_positions = torch.arange(sequence_length, device=features.device).unsqueeze(0)
            q = rope(q, token_positions)
            k = rope(k, token_positions)

        # Append to the cache, if any.
        if past_kv is not None:
            past_k, past_v = past_kv
            length_past = past_k.shape[-2]
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
        else:
            length_past = 0

        length_total = length_past + sequence_length

        if self.attn_implementation == "manual":
            mask = self._causal_mask(sequence_length, length_past, features.device)
            out = scaled_dot_product_attention(q, k, v, mask)
        else:
            out = self._sdpa(q, k, v, sequence_length, length_past, length_total)

        # (..., H, L, D // H) -> (..., L, H, D // H) -> (..., L, D)
        out = out.transpose(-3, -2).reshape(*features.shape[:-1], self.d_model)
        return self.o(out), (k, v)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(..., L, D) -> (..., L, H, D // H)."""
        return x.reshape(*x.shape[:-1], self.num_heads, self.head_dim)

    def _causal_mask(self, length_new: int, length_past: int, device: torch.device) -> torch.Tensor | None:
        """Build a ``(length_new, length_total)`` boolean mask (True = attend)."""
        if not self.use_causal_mask:
            return None
        length_total = length_past + length_new
        mask = torch.ones(length_new, length_total, dtype=torch.bool, device=device)
        # Row i may attend keys 0..(length_past + i).
        return mask.tril(diagonal=length_past)

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        length_new: int,
        length_past: int,
        length_total: int,
    ) -> torch.Tensor:
        """Dispatch to ``F.scaled_dot_product_attention`` (Flash Attention on CUDA)."""
        if not self.use_causal_mask:
            return F.scaled_dot_product_attention(q, k, v)
        if length_past == 0:
            # Full-sequence prefill: the causal mask is square.
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if length_new == 1:
            # Single-token decode: every cached key is in the past, no mask needed.
            return F.scaled_dot_product_attention(q, k, v)
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=self._causal_mask(length_new, length_past, q.device)
        )
