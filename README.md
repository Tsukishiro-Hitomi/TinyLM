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
- **Inference** — top-p / temperature sampling and autoregressive generation
  from a checkpoint.
- **Typed configuration** — `ModelConfig`, `TrainConfig`, and `TokenizerConfig`
  dataclasses replace scattered magic numbers.
- **Tested and linted** — `pytest` and `ruff` are clean, and CI runs both.

Only limited-memory components are exercised here; no trained-model quality or
perplexity result is claimed. The included smoke config is a bounded check, not
a performance recipe.

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

Ideas for extending the project beyond the assignment:

- KV-cache and batched generation for faster inference
- Mixed-precision (bfloat16) training, gradient accumulation, and gradient
  checkpointing
- Grouped-query attention and Flash Attention via
  `torch.nn.functional.scaled_dot_product_attention`
- Streaming dataloader and resume-from-checkpoint training
- Perplexity evaluation plus a small benchmark harness
- Weights & Biases logging and a demo UI

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

The interface conventions follow Stanford CS336, *Language Modeling from
Scratch*.
