# TinyLM — developer tasks. Run `make help` for a summary.
# Requires `uv` (https://docs.astral.sh/uv/). On Windows, run inside Git Bash,
# WSL, or an environment with GNU Make available.

.DEFAULT_GOAL := help
.PHONY: help install lint format test tokenize train generate clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install: ## Create/sync the locked environment
	uv sync

lint: ## Run the linter
	uv run ruff check src scripts tests examples

format: ## Auto-fix lint issues and format
	uv run ruff check --fix src scripts tests examples
	uv run ruff format src scripts tests examples

test: ## Run the test suite
	uv run pytest

tokenize: ## Train the TinyStories BPE tokenizer
	uv run python scripts/train_tokenizer.py --config configs/tokenizer_tinystories.toml

train: ## Run the bounded TinyStories smoke training config
	uv run python scripts/assignment1_experiments.py --config configs/train_tinystories.toml

generate: ## Generate text from a checkpoint
	uv run python scripts/generate.py --config configs/generate_tinystories.toml

clean: ## Remove caches and pytest temp output
	rm -rf .pytest_cache .ruff_cache .pytest_tmp
