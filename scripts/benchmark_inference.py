"""Benchmark KV-cache and attention-implementation speedups.

Reports two views:

1. End-to-end generation (wall clock, includes sampling and per-step sync).
2. Forward-only per-step cost, which isolates the model compute and shows the
   algorithmic benefit of reusing cached keys/values.
"""

from __future__ import annotations

import argparse
import time

import torch

from tinylm import Tokenizer, generate
from _cli_utils import resolve_device
from _model_utils import infer_model_config, load_legacy_state_dict


def _build(state: dict, num_heads: int, context: int, attn: str, device: str):
    config = infer_model_config(state, num_heads, context)
    config.attn_implementation = attn
    model = config.build(device=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_generate(model, tokenizer, prompt: str, tokens: int, device: str, use_cache: bool) -> float:
    torch.manual_seed(0)
    generate(model, tokenizer, prompt, max_new_tokens=8, temperature=0, eos_token=None, device=device, use_cache=use_cache)
    _sync()

    torch.manual_seed(0)
    start = time.perf_counter()
    generate(model, tokenizer, prompt, max_new_tokens=tokens, temperature=0, eos_token=None, device=device, use_cache=use_cache)
    _sync()
    return time.perf_counter() - start


def _time_forward_only(model, vocab_size: int, context: int, steps: int, use_cache: bool) -> float:
    torch.manual_seed(0)
    device = next(model.parameters()).device
    ids = torch.randint(0, vocab_size, (1, context), device=device)
    step_input = torch.zeros(1, 1, dtype=torch.long, device=device)

    with torch.no_grad():
        if use_cache:
            _, cache = model.forward_cached(ids, past_kv=None, start_pos=0)
            _sync()
            start = time.perf_counter()
            for index in range(steps):
                _, cache = model.forward_cached(step_input, past_kv=cache, start_pos=context + index)
            _sync()
        else:
            model(ids)
            _sync()
            start = time.perf_counter()
            for _ in range(steps):
                model(ids)
            _sync()
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--merges", required=True)
    parser.add_argument("--tokens", type=int, default=128, help="Generated tokens for the end-to-end run.")
    parser.add_argument("--context", type=int, default=128, help="Context length for the forward-only run.")
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eos", default="<|endoftext|>")
    args = parser.parse_args()

    device = resolve_device(args.device)
    if device == "cuda":
        device = "cuda:0"

    state = load_legacy_state_dict(args.checkpoint, "cpu")
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, [args.eos])
    prompt = "Once upon a time"
    vocab_size = int(state["embedding.embedding"].shape[0])

    results: dict[tuple[str, bool], tuple[float, float]] = {}
    for attn in ("sdpa", "manual"):
        model = _build(state, args.num_heads, args.max_seq_len, attn, device)
        for use_cache in (True, False):
            elapsed = _time_generate(model, tokenizer, prompt, args.tokens, device, use_cache)
            results[(attn, use_cache)] = (elapsed, args.tokens / elapsed)

    print(f"device={device} tokens={args.tokens} prompt={prompt!r}")
    print("== end-to-end generation ==")
    print(f"{'attn':<8}{'cache':<8}{'time_s':>10}{'tok/s':>12}")
    for (attn, use_cache), (elapsed, tokens_per_second) in results.items():
        print(f"{attn:<8}{str(use_cache):<8}{elapsed:>10.3f}{tokens_per_second:>12.1f}")
    print(f"kv_cache_speedup(sdpa)={results[('sdpa', False)][0] / results[('sdpa', True)][0]:.2f}x")
    print(f"sdpa_vs_manual_speedup(cache)={results[('manual', True)][0] / results[('sdpa', True)][0]:.2f}x")

    # Forward-only: one full pass over `context` tokens vs one cached decode step.
    steps = 100
    forward: dict[bool, float] = {}
    for use_cache in (True, False):
        model = _build(state, args.num_heads, args.max_seq_len, "sdpa", device)
        forward[use_cache] = _time_forward_only(model, vocab_size, args.context, steps, use_cache) / steps
    print(f"\n== forward-only, context={args.context} ==")
    print(f"cached step:     {forward[True] * 1000:.3f} ms/token")
    print(f"full recompute:  {forward[False] * 1000:.3f} ms/token")
    print(f"forward_speedup={forward[False] / forward[True]:.2f}x")


if __name__ == "__main__":
    main()
