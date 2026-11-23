"""Autoregressive text generation with KV-cache support."""

from __future__ import annotations

import torch

from ..tokenization.tokenizer import Tokenizer
from .transformer import TransformerLM


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """Sample one token per batch row from ``logits`` of shape ``(batch, vocab)``.

    ``temperature <= 0`` performs greedy decoding.
    """
    if temperature is None or temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(int(top_k), logits.size(-1))
        threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_index = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        drop = cumulative - probs >= top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_index, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token: str | None = "<|endoftext|>",
    device: torch.device | str | None = None,
    use_cache: bool = True,
) -> str:
    """Generate a continuation of ``prompt``.

    With ``use_cache=True`` the prompt is prefilled once and every new token is
    decoded against the cached keys/values in O(L) rather than recomputing the
    full prefix. The output is identical to the no-cache path for greedy decoding.
    """
    device = torch.device(device) if device is not None else next(model.parameters()).device
    eos_id = tokenizer.encode(eos_token)[0] if eos_token else None

    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [eos_id] if eos_id is not None else [0]
    generated: list[int] = []

    if use_cache:
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        logits, cache = model.forward_cached(input_ids, past_kv=None, start_pos=0)
        position = len(prompt_ids)
        for step in range(max_new_tokens):
            next_token = sample_next_token(logits[:, -1, :], temperature, top_k, top_p).item()
            generated.append(next_token)
            if eos_id is not None and next_token == eos_id:
                break
            if step == max_new_tokens - 1:
                break
            step_input = torch.tensor([[next_token]], dtype=torch.long, device=device)
            logits, cache = model.forward_cached(step_input, past_kv=cache, start_pos=position)
            position += 1
    else:
        ids = list(prompt_ids)
        for _ in range(max_new_tokens):
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            logits = model(input_ids)
            next_token = sample_next_token(logits[:, -1, :], temperature, top_k, top_p).item()
            generated.append(next_token)
            ids.append(next_token)
            if eos_id is not None and next_token == eos_id:
                break

    return tokenizer.decode(prompt_ids + generated)
