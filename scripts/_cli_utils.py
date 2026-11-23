"""Small, dependency-free helpers shared by command-line scripts."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Any

import torch


def resolve_device(requested: str) -> str:
    """Resolve ``auto`` without assuming a particular accelerator index."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def config_defaults(path: str | None, section: str) -> dict[str, Any]:
    """Load defaults from one table in a TOML file."""
    if path is None:
        return {}
    config_path = Path(path)
    try:
        with config_path.open("rb") as config_file:
            document = tomllib.load(config_file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise argparse.ArgumentTypeError(f"cannot read config {config_path}: {error}") from error
    values = document.get(section)
    if not isinstance(values, dict):
        raise argparse.ArgumentTypeError(f"config {config_path} must contain a [{section}] table")
    return values


def config_path_from_argv(description: str) -> str | None:
    """Read only --config so its values can become argparse defaults."""
    parser = argparse.ArgumentParser(add_help=False, description=description)
    parser.add_argument("--config")
    known, _ = parser.parse_known_args()
    return known.config
