# TinyLM

A compact, from-scratch implementation of a modern Transformer language model —
byte-level BPE tokenizer, RoPE attention, SwiGLU feed-forward, RMSNorm, a
training loop with AdamW and a cosine schedule, and checkpointed text
generation. The whole stack is implemented directly in PyTorch with no
high-level model framework.

The code was originally developed while working through Stanford CS336
(assignment 1) and has since been restructured into a small, reusable library.

## Highlights

- **Byte-level BPE tokenizer** with parallel pre-tokenization and an
  incremental pair-frequency merge loop. On TinyStories (10k vocab) the
  optimized merge stage runs ~24× faster than the naive rescan version
  (778 s → 32.5 s; 969 s → 130 s end to end).
- **Modern Transformer** — rotary position embeddings, grouped multi-head
  self-attention, SwiGLU, pre-norm RMSNorm, weight tying.
- **Training utilities** — AdamW, SGD, cosine learning-rate schedule with
  warmup, gradient clipping, checkpoint save/load, CSV metric logging, and loss
  plots.
- **Inference** — KV-cache incremental decoding, Flash Attention via
  `F.scaled_dot_product_attention` (with a manual fallback for parity tests),
  and temperature / top-k / top-p sampling.
- **Evaluation** — tokenizer-independent bits-per-byte (BPB), generation
  diversity (distinct-n and repetition), and an inference benchmark.
- **Typed configuration** — `ModelConfig`, `TrainConfig`, and `TokenizerConfig`
  dataclasses replace scattered magic numbers.
- **Tested and linted** — 49 tests pass and `ruff` is clean; CI runs both.

Results for the 22.7M-parameter model on TinyStories: **val loss 1.375, PPL 3.96,
BPB 0.4996**, with full learning-rate / batch-size / ablation sweeps in
[docs/RESULTS.md](docs/RESULTS.md).

## Project layout

| Path | Purpose |
| --- | --- |
| `src/tinylm/` | The library: config, model, tokenization, training |
| `src/tinylm/model/` | Attention, layers, Transformer, inference |
| `src/tinylm/tokenization/` | Byte-level BPE training and `Tokenizer` |
| `src/tinylm/training/` | Data batching, losses, optimizers, trainer, utilities |
| `src/tinylm/config.py` | Typed hyperparameter dataclasses and model factory |
| `scripts/` | Command-line entry points (tokenize, train, generate, plot) |
| `configs/` | Small TOML examples for each CLI |
| `examples/` | Runnable end-to-end API demo |
| `notebooks/` | Quickstart notebook |
| `tests/` | Unit and snapshot tests |
| `docs/` | Command reference and CS336 notes |

## Requirements

- Python `>=3.12,<3.14`
- [`uv`](https://docs.astral.sh/uv/)
- A compatible PyTorch device; CPU works for tests and small demos

Create or sync the locked environment:

```sh
uv sync
```

## Quick start

Run the self-contained demo (builds a model, runs a forward pass, and trains a
tiny tokenizer on a bundled fixture):

```sh
uv run python examples/quickstart.py
```

CLI entry points:

```sh
uv run python scripts/train_tokenizer.py --help
uv run python scripts/assignment1_experiments.py --help
uv run python scripts/generate.py --help
```

Config-driven examples (each requires the named data/checkpoint):

```sh
uv run python scripts/train_tokenizer.py --config configs/tokenizer_tinystories.toml
uv run python scripts/assignment1_experiments.py --config configs/train_tinystories.toml
uv run python scripts/generate.py --config configs/generate_tinystories.toml
```

All maintained entry points default to `--device auto`, which selects CUDA, then
MPS, then CPU. See [docs/running.md](docs/running.md) for data preparation and
the full command reference.

## Tests and checks

```sh
uv run pytest
uv run ruff check src scripts tests
```

`make test` and `make lint` wrap the same commands.

## Training results

A 22.7M-parameter model (512 d_model, 16 heads, 4 layers) trained on TinyStories
reaches **val loss 1.375 (PPL 3.96)** after 40k steps on a single A800. Full
sweeps, ablation tables, and figures are in [docs/RESULTS.md](docs/RESULTS.md).

| Experiment | Config | Val loss | PPL |
| --- | --- | ---: | ---: |
| Baseline | lr 1e-3, batch 32, 40k steps | **1.375** | **3.96** |
| LR sweep | best lr 1e-3 | 1.378 | 3.97 |
| Ablation | post-norm / SiLU / no-RoPE | 1.38–1.43 | 3.97–4.18 |
| OpenWebText | 32k vocab, baseline | 3.939 | 51.36 |

Sample continuation from `checkpoint40000` (`temperature=0.8, top_p=0.9`):

> Once upon a time, there was a little girl named Lily. She loved to eat apples.
> One day, she went to the park with her mom. They saw a big tree with lots of
> apples.

![Training loss curve](docs/figures/main_loss_curve.png)

Reproduce the run summary from any directory of `metrics.csv` files:

```sh
uv run python scripts/summarize_runs.py --runs-dir runs
```

## Measured results (tokenizer)

From the original experiment notes in [docs/cs336-notes.txt](docs/cs336-notes.txt):

| Run | Naive | Optimized |
| --- | ---: | ---: |
| TinyStories BPE (10k vocab) | 969.6 s | 130.0 s |
| — merge stage only | 778.2 s | 32.5 s |

Tokenizer compression (bytes/token): OpenWebText 4.68, TinyStories 4.15, and
cross-domain (TinyStories tokenizer on OpenWebText) 3.19. Encoding throughput is
roughly 0.5 MB/s in pure Python.

## Roadmap

- Batched generation plus `torch.compile`/CUDA graphs — the 22M model at batch 1
  is CPU-dispatch-bound, so these are needed to expose the KV-cache compute win
- Mixed-precision (bfloat16) training, gradient accumulation, gradient checkpointing
- Grouped-query attention
- Streaming dataloader and resume-from-checkpoint training
- LLM-as-judge generation scoring and a broader benchmark harness
- Weights & Biases logging and a demo UI

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

The interface conventions follow Stanford CS336, *Language Modeling from
Scratch*.
