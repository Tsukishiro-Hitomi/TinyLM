# Running the project

Run commands from the repository root. All examples use `uv run`, so the
project environment described by `pyproject.toml` and `uv.lock` is used.

## Data flow

1. Place the TinyStories or OpenWebText text files in `data/`.
2. Train a tokenizer with `scripts/train_tokenizer.py`.
3. Encode text into the `uint16` `.bin` files expected by the training scripts.
4. Train with `scripts/assignment1_experiments.py`.
5. Generate text from a saved checkpoint with `scripts/generate.py`.

Downloaded data, tokenizer artifacts, checkpoints, metrics, and generated text
are local outputs and are ignored by Git.

## Tokenizer

The original positional dataset argument remains supported:

```sh
uv run python scripts/train_tokenizer.py tinystories
```

The equivalent config-driven command is:

```sh
uv run python scripts/train_tokenizer.py --config configs/tokenizer_tinystories.toml
```

Use `--input-path`, `--vocab-size`, and `--output-dir` to override individual
values. Tokenizer training reads the full input corpus and can be expensive.

## Encoding

After both preset tokenizers have been trained and both datasets downloaded,
the legacy encoder creates the four training/validation binary files:

```sh
uv run python scripts/encode_text.py
```

`scripts/owt_fast_encode.py` is an OpenWebText-specific compatibility utility.
It validates a `tiktoken` encoding against the project tokenizer before writing
temporary buffered outputs.

## Training and evaluation

Inspect all options without starting a run:

```sh
uv run python scripts/assignment1_experiments.py --help
```

A bounded smoke configuration is included. It still requires encoded data and
performs real training; increase its limits deliberately for a full run.

```sh
uv run python scripts/assignment1_experiments.py --config configs/train_tinystories.toml
```

Evaluate an existing checkpoint:

```sh
uv run python scripts/assignment1_experiments.py --evaluate --checkpoint PATH --dataset tinystories
```

The legacy `scripts/train_transformer.py` entry point remains available and now
accepts path, device, and run-size flags. The experiment runner is the preferred
entry point because it records a compact metrics CSV and supports evaluation.

## Generation

```sh
uv run python scripts/generate.py --config configs/generate_tinystories.toml
```

The old `assignment1_generate.py` and `assignment1_generate_owt.py` names remain
compatible and forward to the same CLI with different dataset presets.

Every training and generation CLI defaults to `--device auto`: CUDA when
available, then Apple MPS when available, otherwise CPU. Select a device
explicitly with values such as `--device cpu`, `--device cuda`, or
`--device cuda:0`; no GPU index is assumed by default.
