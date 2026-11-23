"""Measure generation diversity and repetition from a checkpoint.

Reports corpus-level distinct-1/2/3 and a repetition rate, and writes every
sample to JSONL so the generations can be scored by an LLM-as-judge later.

Usage:

    uv run python scripts/eval_generation.py --checkpoint ... --vocab ... --merges ... \
        --output docs/generations.jsonl --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tinylm import Tokenizer, generate
from _cli_utils import resolve_device
from _model_utils import infer_model_config, load_legacy_state_dict

DEFAULT_PROMPTS = [
    "Once upon a time",
    "There was a little",
    "One day, the",
    "The sun was",
    "A small dog named",
    "Tom and his friend",
    "The girl opened the",
    "In the forest, there",
]


def distinct_n(texts: list[str], n: int) -> float:
    grams: list[tuple[str, ...]] = []
    for text in texts:
        words = text.split()
        grams.extend(tuple(words[index : index + n]) for index in range(len(words) - n + 1))
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--merges", required=True)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=3, help="Samples per prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=336)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="docs/generations.jsonl")
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

    torch.manual_seed(args.seed)
    records = []
    for prompt in DEFAULT_PROMPTS:
        for _ in range(args.num_samples):
            text = generate(
                model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token=args.eos,
                device=device,
                use_cache=True,
            )
            continuation = text[len(prompt):] if text.startswith(prompt) else text
            records.append({"prompt": prompt, "text": text, "continuation": continuation})

    continuations = [record["continuation"] for record in records]

    print(f"samples={len(records)} prompts={len(DEFAULT_PROMPTS)}")
    for n in (1, 2, 3):
        print(f"distinct-{n}={distinct_n(continuations, n):.4f}")
    print(f"repetition-4={1.0 - distinct_n(continuations, 4):.4f}")
    print(f"unique_samples={len(set(continuations)) / len(continuations):.4f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records), encoding="utf-8")
    print(f"wrote {len(records)} samples to {output}")


if __name__ == "__main__":
    main()
