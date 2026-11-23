"""Configurable text generation for the Assignment 1 models."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tinylm import RoPE, Tokenizer, TransformerLM
from _cli_utils import config_defaults, config_path_from_argv, resolve_device


PRESETS = {
    "tinystories": {
        "vocab_size": 10000,
        "vocab_path": "artifacts/tokenizer/tinystories_vocab.pkl",
        "merges_path": "artifacts/tokenizer/tinystories_merges.pkl",
        "checkpoint": "runs/lr_sweep_0.001_seed336/checkpoint_final",
        "prompt": "Once upon a time",
        "output": "runs/learning_rate_comparison/generated_seed336.txt",
    },
    "owt": {
        "vocab_size": 32000,
        "vocab_path": "artifacts/tokenizer/owt_vocab.pkl",
        "merges_path": "artifacts/tokenizer/owt_merges.pkl",
        "checkpoint": "runs/owt_baseline_seed336/checkpoint_final",
        "prompt": "The researchers discovered that",
        "output": "runs/owt_baseline_seed336/generated_seed336.txt",
    },
}


def build_parser(default_dataset: str) -> argparse.ArgumentParser:
    description = "Generate text from an Assignment 1 Transformer checkpoint."
    config_path = config_path_from_argv(description)
    defaults = config_defaults(config_path, "generation")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", help="TOML file containing a [generation] table.")
    parser.add_argument("--dataset", choices=PRESETS, default=default_dataset)
    parser.add_argument("--checkpoint", help="Checkpoint path (otherwise uses the dataset preset).")
    parser.add_argument("--vocab-path", help="Vocabulary pickle path (otherwise uses the preset).")
    parser.add_argument("--merges-path", help="Merges pickle path (otherwise uses the preset).")
    parser.add_argument("--prompt", help="Prompt text (otherwise uses the preset).")
    parser.add_argument("--output", help="Output text path (otherwise uses the preset).")
    parser.add_argument("--device", default="auto", help="Torch device (default: auto; e.g. cpu, cuda, cuda:0, mps).")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=336)
    parser.set_defaults(**defaults)
    return parser


def main(default_dataset: str = "tinystories") -> None:
    args = build_parser(default_dataset).parse_args()
    preset = PRESETS[args.dataset]
    device = resolve_device(args.device)
    checkpoint = args.checkpoint or preset["checkpoint"]
    vocab_path = args.vocab_path or preset["vocab_path"]
    merges_path = args.merges_path or preset["merges_path"]
    prompt = args.prompt or preset["prompt"]
    output_path = Path(args.output or preset["output"])

    if args.temperature <= 0:
        raise ValueError("--temperature must be greater than zero")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tok = Tokenizer.from_files(vocab_path, merges_path, ["<|endoftext|>"])
    rope = RoPE(theta=10000, d_k=32, max_seq_len=256, device=device)
    model = TransformerLM(
        d_model=512, num_heads=16, d_ff=1344, rope=rope,
        vocab_size=preset["vocab_size"], num_layers=4, device=device,
    ).to(device)
    step, weights, _ = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(weights)
    model.eval()
    ids = tok.encode(prompt)
    initial_length = len(ids)
    eos = tok.encode("<|endoftext|>")[0]
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            x = torch.tensor([ids[-256:]], dtype=torch.long, device=device)
            probabilities = torch.softmax(model(x)[0, -1] / args.temperature, dim=-1)
            sorted_probabilities, order = probabilities.sort(descending=True)
            remove = sorted_probabilities.cumsum(dim=-1) - sorted_probabilities >= args.top_p
            sorted_probabilities[remove] = 0
            sorted_probabilities /= sorted_probabilities.sum()
            token = order[torch.multinomial(sorted_probabilities, 1)].item()
            ids.append(token)
            if token == eos:
                break

    output = tok.decode(ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output + "\n", encoding="utf-8")
    print(
        f"step={step} generated_tokens={len(ids) - initial_length} "
        f"temperature={args.temperature} top_p={args.top_p} device={device}"
    )
    print(output)


if __name__ == "__main__":
    main()
