"""Legacy training entry point with configurable paths and device selection."""

import argparse

import numpy as np

from tinylm import AdamW, RoPE, TransformerLM, cross_entropy, train
from _cli_utils import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the legacy TinyStories Transformer configuration.")
    parser.add_argument("--train-data", default="data/tinystories_train.bin")
    parser.add_argument("--valid-data", default="data/tinystories_valid.bin")
    parser.add_argument("--device", default="auto", help="Torch device (default: auto; e.g. cpu, cuda, cuda:0, mps).")
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--evaluate-every", type=int, default=200)
    args = parser.parse_args()

    device = resolve_device(args.device)
    train_data = np.memmap(args.train_data, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.valid_data, dtype=np.uint16, mode="r")
    d_model = 512
    num_heads = 16
    rope = RoPE(theta=10000, d_k=d_model // num_heads, max_seq_len=args.context_length, device=device)
    model = TransformerLM(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=1344,
        rope=rope,
        vocab_size=10000,
        num_layers=4,
        device=device,
    )
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    train(
        train_data,
        val_data,
        args.batch_size,
        args.context_length,
        device,
        args.steps,
        model,
        cross_entropy,
        optimizer,
        args.save_every,
        args.evaluate_every,
    )


if __name__ == "__main__":
    main()
