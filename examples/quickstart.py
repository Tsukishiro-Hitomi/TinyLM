"""End-to-end TinyLM quickstart.

Builds a small TransformerLM from a config, runs a forward pass, trains a tiny
BPE tokenizer on a fixture, and shows a tokenizer round-trip. Generation from a
randomly initialized model is not included because the output would be
meaningless without training; see ``scripts/generate.py`` for checkpoint-based
generation.

Run:

    uv run python examples/quickstart.py
"""

from __future__ import annotations

import torch

from tinylm import ModelConfig, Tokenizer, run_bpe_train


def main() -> None:
    torch.manual_seed(0)

    # 1. Build a language model from a typed config.
    config = ModelConfig(
        d_model=128,
        num_heads=4,
        d_ff=512,
        vocab_size=512,
        num_layers=2,
        max_seq_len=128,
    )
    model = config.build(device="cpu")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}")

    # 2. Forward pass on random token ids.
    tokens = torch.randint(0, config.vocab_size, (2, 32))
    logits = model(tokens)
    print(f"input shape:      {tuple(tokens.shape)}")
    print(f"logits shape:     {tuple(logits.shape)}")

    # 3. Train a tiny BPE tokenizer on a small fixture corpus.
    vocab, merges = run_bpe_train(
        "tests/fixtures/corpus.en",
        vocab_size=300,
        special_tokens=["<|endoftext|>"],
    )
    tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])

    text = "Hello, world! This is TinyLM."
    ids = tokenizer.encode(text)
    print(f"encoded:          {ids}")
    print(f"decoded:          {tokenizer.decode(ids)!r}")
    assert tokenizer.decode(ids) == text


if __name__ == "__main__":
    main()
