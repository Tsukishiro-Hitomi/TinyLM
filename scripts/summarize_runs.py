"""Summarize training runs into a markdown table of final losses.

Each run directory is expected to contain a ``metrics.csv`` with columns such
as ``step,train_loss,val_loss,lr``. The final value of each series is reported
together with the token-level perplexity (``exp(val_loss)``).

Usage:

    uv run python scripts/summarize_runs.py --runs-dir runs
    uv run python scripts/summarize_runs.py --runs-dir runs --json out.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read_metrics(path: Path) -> list[dict[str, float]]:
    """Read a metrics CSV tolerantly, skipping unparsable rows/fields."""
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return rows
        for raw in reader:
            record: dict[str, float] = {}
            for key, value in raw.items():
                if key is None or value is None or value == "":
                    continue
                try:
                    record[key] = float(value)
                except ValueError:
                    continue
            if record:
                rows.append(record)
    return rows


def last_value(rows: list[dict[str, float]], key: str) -> float | None:
    for record in reversed(rows):
        if key in record:
            return record[key]
    return None


def classify(name: str) -> str:
    if name.startswith("lr_sweep"):
        return "lr_sweep"
    if name.startswith("batch"):
        return "batch_sweep"
    if name.startswith("ablation"):
        return "ablation"
    if name.startswith("owt"):
        return "owt"
    if name[:8].isdigit() and name[8:9] == "-":
        return "baseline"
    return "other"


def collect(runs_dir: Path) -> list[dict]:
    runs = []
    for run_dir in sorted(runs_dir.iterdir()):
        metrics = run_dir / "metrics.csv"
        if not metrics.is_file():
            continue
        rows = read_metrics(metrics)
        if not rows:
            continue
        val_loss = last_value(rows, "val_loss")
        runs.append(
            {
                "run": run_dir.name,
                "group": classify(run_dir.name),
                "steps": int(rows[-1].get("step", 0)),
                "train_loss": last_value(rows, "train_loss"),
                "val_loss": val_loss,
                "ppl": math.exp(val_loss) if val_loss is not None else None,
            }
        )
    return runs


def markdown_table(runs: list[dict], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"

    def cell(run: dict, key: str) -> str:
        value = run.get(key)
        if value is None:
            return "-"
        if key in ("val_loss", "train_loss"):
            return f"{value:.3f}"
        if key == "ppl":
            return f"{value:.2f}"
        if key == "steps":
            return f"{value:,}"
        return str(value)

    body = ["| " + " | ".join(cell(run, c) for c in columns) + " |" for run in runs]
    return "\n".join([header, divider, *body])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs", help="Directory containing run subdirectories.")
    parser.add_argument("--json", help="Optional path to also write the raw summary as JSON.")
    args = parser.parse_args()

    runs = collect(Path(args.runs_dir))
    if not runs:
        raise SystemExit(f"no runs with metrics.csv found under {args.runs_dir}")

    runs.sort(key=lambda r: (r["val_loss"] is None, r["val_loss"] or 0.0))

    columns = ["group", "run", "steps", "train_loss", "val_loss", "ppl"]
    print(markdown_table(runs, columns))

    if args.json:
        Path(args.json).write_text(json.dumps(runs, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
