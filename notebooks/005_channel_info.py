"""Inspect channel metadata for EEG releases."""

# %%
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from rich.console import Console
from rich.table import Table

from cerebro.data import load_challenge_dataset, load_chs_info

console = Console()

# %%
parser = ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="RestingState")
parser.add_argument("--release", type=str, default="R1")
parser.add_argument("--dataset-variant", choices=["mini", "full"], default="mini")
parser.add_argument("--data-root", type=Path, default=Path("data"))
parser.add_argument("--full-subdir", type=str, default="full")
args = parser.parse_args()

# %%
console.rule("Channel metadata")
console.print(f"Task: [cyan]{args.task}[/cyan], Release: [cyan]{args.release}[/cyan]")
console.print(f"Variant: [cyan]{args.dataset_variant}[/cyan]")

try:
    ds = load_challenge_dataset(
        task=args.task,
        release=args.release,
        variant=args.dataset_variant,
        cache_dir=args.data_root,
        full_subdir=args.full_subdir,
    )
except Exception as exc:
    console.print(f"[red]Failed to load dataset: {exc}")
    raise SystemExit(1)

if not ds.datasets:
    console.print("[yellow]No recordings found.[/yellow]")
    raise SystemExit(0)

chs = load_chs_info(
    args.task,
    args.release,
    variant=args.dataset_variant,
    cache_dir=args.data_root,
    full_subdir=args.full_subdir,
)
raw = ds.datasets[0].raw
console.print(f"Total channels: [bold]{len(chs)}[/bold]")

# Display first few entries
preview = Table(title="Sample channel info", show_lines=True)
preview.add_column("Index", style="cyan")
preview.add_column("Name")
preview.add_column("Loc (first 3 components)")
preview.add_column("Kind")

for idx, ch in enumerate(chs[:10]):
    loc = ch.get("loc", [])
    loc_str = "None" if loc is None else ", ".join(f"{v:.3f}" for v in loc[:3])
    preview.add_row(str(idx), ch.get("ch_name", ""), loc_str, str(ch.get("kind", "")))

console.print(preview)

# Summary statistics
n_missing = sum(1 for ch in chs if ch.get("loc") is None or all(v == 0 for v in ch.get("loc", [])))
console.print(f"Channels with missing/zero loc: [red]{n_missing}[/red]")
