"""Minimal smoke test: build, forward, and (optionally) load a checkpoint.

Verifies that the library imports, the requested device is usable, and a
checkpoint produced by the earlier implementation loads into the current model.

Usage:

    uv run python scripts/smoke_test.py
    uv run python scripts/smoke_test.py --device cuda:0 \
        --checkpoint /path/to/checkpoint --vocab /path/to/vocab.pkl --merges /path/to/merges.pkl
"""

from __future__ import annotations

import argparse

import torch

from tinylm import ModelConfig, Tokenizer
from _cli_utils import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", help="Optional checkpoint from the legacy implementation.")
    parser.add_argument("--vocab", help="Tokenizer vocabulary pickle (with --checkpoint).")
    parser.add_argument("--merges", help="Tokenizer merges pickle (with --checkpoint).")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"device: {device}")

    if args.checkpoint:
        config = ModelConfig(vocab_size=10_000, max_seq_len=256)
        model = config.build(device=device)
        payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state = payload[1] if isinstance(payload, tuple) else payload
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise SystemExit(f"state_dict mismatch: missing={missing} unexpected={unexpected}")
        print(f"loaded checkpoint: {args.checkpoint}")
    else:
        config = ModelConfig(d_model=128, num_heads=4, d_ff=512, num_layers=2, vocab_size=512, max_seq_len=128)
        model = config.build(device=device)

    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {params:,}")

    tokens = torch.randint(0, config.vocab_size, (2, 16), device=device)
    with torch.no_grad():
        logits = model(tokens)
    print(f"input:  {tuple(tokens.shape)}")
    print(f"logits: {tuple(logits.shape)}")
    assert torch.isfinite(logits).all(), "non-finite logits"

    if args.checkpoint and args.vocab and args.merges:
        tokenizer = Tokenizer.from_files(args.vocab, args.merges, ["<|endoftext|>"])
        ids = tokenizer.encode("Once upon a time")
        print(f"tokenizer sample: {ids[:8]} -> {tokenizer.decode(ids)!r}")

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
