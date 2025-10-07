#!/usr/bin/env python
"""Precompute fixed-length windows for selected releases/tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.progress import track

import torch
from braindecode.datasets import BaseConcatDataset

from cerebro.data import load_challenge_dataset
from cerebro.constants import MINI_DATASET_ROOT, SUBJECTS_TO_REMOVE, SFREQ
from cerebro.preprocessing import (
    filter_by_anchor,
    prepare_dataset,
    create_single_windows,
)
from braindecode.preprocessing import create_fixed_length_windows

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-variant",
        choices=["mini", "full"],
        default="full",
        help="Which dataset variant to cache (default: full)",
    )
    parser.add_argument(
        "--releases",
        nargs="+",
        default=[f"R{i}" for i in range(1, 12)],
        help="Releases to process",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "RestingState",
            "surroundSupp",
            "contrastChangeDetection",
        ],
        help="Tasks to process",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MINI_DATASET_ROOT / "cached_windows",
        help="Directory to store cached windows",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cached files",
    )
    return parser.parse_args()


def cache_windows(dataset, task: str, release: str, output_path: Path, overwrite: bool = False) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    cache_file = output_path / f"windows_{task}_{release}.pt"
    meta_file = output_path / f"windows_{task}_{release}.json"
    if cache_file.exists() and not overwrite:
        console.log(f"Skipping {task}/{release}; cache exists")
        return

    recordings = []
    for base_ds in dataset.datasets:
        subj = str(base_ds.description.get("subject", ""))
        if subj in SUBJECTS_TO_REMOVE:
            continue
        base_ds.description.loc["subject"] = subj
        recordings.append(base_ds)

    dataset = BaseConcatDataset(recordings)

    if task == "contrastChangeDetection":
        dataset = prepare_dataset(dataset, epoch_length=2.0)
        dataset = filter_by_anchor(dataset)
        windows = create_single_windows(
            dataset,
            window_len=2.0,
            shift_after_stim=0.5,
            epoch_len_s=2.0,
            sfreq=SFREQ,
        )
    else:
        windows = create_fixed_length_windows(
            dataset,
            window_size_samples=int(4.0 * SFREQ),
            window_stride_samples=int(2.0 * SFREQ),
            drop_last_window=True,
        )

    split = windows.split("subject")
    kept = []
    for subject, ds in split.items():
        subj = str(subject)
        ds.description.loc["subject"] = subj
        ds.description.loc["task"] = task
        ds.description.loc["release"] = release
        kept.append(ds)

    windows = BaseConcatDataset(kept)

    torch.save(windows, cache_file)
    meta = {
        "task": task,
        "release": release,
        "window_size": [2.0, SFREQ],
        "n_windows": len(windows)
    }
    meta_file.write_text(json.dumps(meta, indent=2))
    console.log(f"Cached windows for {task}/{release} at {cache_file}")


def main() -> None:
    args = parse_args()
    console.rule("Window caching")
    console.print(f"Variant: [bold]{args.dataset_variant}[/bold]")
    console.print(f"Output dir: [green]{args.output_dir}[/green]")

    for release in args.releases:
        for task in track(args.tasks, description=f"Release {release}"):
            try:
                dataset = load_challenge_dataset(
                    task=task,
                    release=release,
                    variant=args.dataset_variant,
                    cache_dir=MINI_DATASET_ROOT,
                    full_subdir="full",
                )
            except Exception as exc:
                console.log(f"[yellow]Skipping {task}/{release}: {exc}")
                continue
            cache_windows(
                dataset,
                task,
                release,
                args.output_dir / args.dataset_variant,
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main()
