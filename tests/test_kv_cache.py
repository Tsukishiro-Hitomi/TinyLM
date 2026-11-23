"""Tests for KV-cache equivalence and attention-implementation parity."""

import torch

from tinylm import ModelConfig, Tokenizer, generate


def _small(attn_implementation: str = "sdpa") -> ModelConfig:
    return ModelConfig(
        d_model=64,
        num_heads=4,
        d_ff=128,
        num_layers=2,
        vocab_size=100,
        max_seq_len=32,
        attn_implementation=attn_implementation,
    )


def test_kv_cache_matches_full_forward() -> None:
    torch.manual_seed(0)
    model = _small().build(device="cpu").eval()
    ids = torch.randint(0, 100, (2, 12))
    with torch.no_grad():
        full = model(ids)
        logits, cache = model.forward_cached(ids[:, :1], past_kv=None, start_pos=0)
        pieces = [logits]
        for pos in range(1, ids.shape[1]):
            logits, cache = model.forward_cached(ids[:, pos : pos + 1], past_kv=cache, start_pos=pos)
            pieces.append(logits)
        stepwise = torch.cat(pieces, dim=1)
    torch.testing.assert_close(full, stepwise, rtol=1e-4, atol=1e-4)


def test_sdpa_matches_manual() -> None:
    torch.manual_seed(0)
    sdpa = _small("sdpa").build(device="cpu").eval()
    manual = _small("manual").build(device="cpu").eval()
    manual.load_state_dict(sdpa.state_dict())
    ids = torch.randint(0, 100, (2, 8))
    with torch.no_grad():
        torch.testing.assert_close(sdpa(ids), manual(ids), rtol=1e-4, atol=1e-4)


def test_greedy_cached_generation_matches_no_cache() -> None:
    torch.manual_seed(0)
    # A byte-level tokenizer needs the full 256-entry base vocabulary.
    model = ModelConfig(
        d_model=64,
        num_heads=4,
        d_ff=128,
        num_layers=2,
        vocab_size=256,
        max_seq_len=32,
    ).build(device="cpu").eval()
    vocab = {i: bytes([i]) for i in range(256)}
    tokenizer = Tokenizer(vocab, [], [])

    cached = generate(model, tokenizer, "hello", max_new_tokens=8, temperature=0, eos_token=None, use_cache=True)
    no_cache = generate(model, tokenizer, "hello", max_new_tokens=8, temperature=0, eos_token=None, use_cache=False)
    assert cached == no_cache
