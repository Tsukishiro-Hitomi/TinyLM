"""Configurable text generation for the tinylm models."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tinylm import Tokenizer, generate
from _cli_utils import config_defaults, config_path_from_argv, resolve_device
from _model_utils import infer_model_config, load_legacy_state_dict


PRESETS = {
    "tinystories": {
        "vocab_path": "artifacts/tokenizer/tinystories_vocab.pkl",
        "merges_path": "artifacts/tokenizer/tinystories_merges.pkl",
        "checkpoint": "runs/lr_sweep_0.001_seed336/checkpoint_final",
        "prompt": "Once upon a time",
        "output": "runs/learning_rate_comparison/generated_seed336.txt",
    },
    "owt": {
        "vocab_path": "artifacts/tokenizer/owt_vocab.pkl",
        "merges_path": "artifacts/tokenizer/owt_merges.pkl",
        "checkpoint": "runs/owt_baseline_seed336/checkpoint_final",
        "prompt": "The researchers discovered that",
        "output": "runs/owt_baseline_seed336/generated_seed336.txt",
    },
}


def build_parser(default_dataset: str) -> argparse.ArgumentParser:
    description = "Generate text from a tinylm Transformer checkpoint."
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
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=336)
    parser.add_argument("--no-cache", action="store_true", help="Disable the KV cache (slower baseline).")
    parser.set_defaults(**defaults)
    return parser


def main(default_dataset: str = "tinystories") -> None:
    args = build_parser(default_dataset).parse_args()
    preset = PRESETS[args.dataset]
    device = resolve_device(args.device)
    if device == "cuda":
        device = "cuda:0"
    checkpoint = args.checkpoint or preset["checkpoint"]
    vocab_path = args.vocab_path or preset["vocab_path"]
    merges_path = args.merges_path or preset["merges_path"]
    prompt = args.prompt or preset["prompt"]
    output_path = Path(args.output or preset["output"])

    if args.temperature <= 0:
        raise ValueError("--temperature must be greater than zero")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")

    state = load_legacy_state_dict(checkpoint, "cpu")
    config = infer_model_config(state, args.num_heads, args.context)
    model = config.build(device=device)
    model.load_state_dict(state)
    model.eval()

    tokenizer = Tokenizer.from_files(vocab_path, merges_path, ["<|endoftext|>"])

    torch.manual_seed(args.seed)
    text = generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token="<|endoftext|>",
        device=device,
        use_cache=not args.no_cache,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text + "\n", encoding="utf-8")
    print(
        f"generated_tokens={args.max_new_tokens} temperature={args.temperature} "
        f"top_p={args.top_p} device={device} cache={not args.no_cache}"
    )
    print(text)


if __name__ == "__main__":
    main()
