"""Training loss functions."""

import torch
from einops import rearrange

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # inputs: (batch, context_length, vocab_size)
    # targets: (batch, context_length)
    logits = rearrange(inputs, "... v -> (...) v")
    targets = rearrange(targets, "... -> (...)")
    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (torch.logsumexp(logits, dim=-1) - target_logits).mean()
