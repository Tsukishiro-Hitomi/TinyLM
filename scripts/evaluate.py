"""Compute validation loss, perplexity, and bits-per-byte (BPB).

BPB is tokenizer-independent (it divides the negative log-likelihood by the
number of UTF-8 bytes in the targets), so runs that use different vocabularies
are directly comparable.

Usage:

    uv run python scripts/evaluate.py --checkpoint ... --vocab ... --merges ... \
        --data data/tinystories_valid.bin --device cuda:0
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from tinylm import Tokenizer, cross_entropy, get_batch
from _cli_utils import resolve_device
from _model_utils import infer_model_config, load_legacy_state_dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--merges", required=True)
    parser.add_argument("--data", required=True, help="Tokenized .bin validation file.")
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eos", default="<|endoftext|>")
    args = parser.parse_args()

    device = resolve_device(args.device)
    if device == "cuda":
        device = "cuda:0"

    state = load_legacy_state_dict(args.checkpoint, "cpu")
    config = infer_model_config(state, args.num_heads, args.context)
    model = config.build(device=device)
    model.load_state_dict(state)
    model.eval()

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, [args.eos])
    byte_lengths = np.array(
        [len(tokenizer.vocab.get(index, b"")) for index in range(config.vocab_size)], dtype=np.int64
    )

    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    torch.manual_seed(336)

    total_nll = 0.0
    total_tokens = 0
    total_bytes = 0
    with torch.no_grad():
        for _ in range(args.batches):
            x, y = get_batch(data, args.batch_size, args.context, device)
            loss = cross_entropy(model(x), y)
            count = y.numel()
            total_nll += loss.item() * count
            total_tokens += count
            total_bytes += int(byte_lengths[y.cpu().numpy()].sum())

    mean_nll = total_nll / total_tokens
    bpb = total_nll / math.log(2) / total_bytes
    print(f"data={args.data}")
    print(f"batches={args.batches} tokens={total_tokens} bytes={total_bytes}")
    print(f"val_loss={mean_nll:.4f} ppl={math.exp(mean_nll):.4f} bpb={bpb:.4f}")


if __name__ == "__main__":
    main()
