"""Subject overlap analysis across releases and tasks."""

# %%
from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table

from cerebro.data import load_challenge_dataset
from cerebro.constants import SUBJECTS_TO_REMOVE

# %%
parser = ArgumentParser(description=__doc__)
parser.add_argument(
    "--dataset-variant",
    choices=["mini", "full"],
    default="mini",
    help="Dataset variant to inspect (default: mini)",
)
parser.add_argument(
    "--data-root",
    type=Path,
    default=Path("data"),
    help="Cache directory root (default: ./data)",
)
parser.add_argument(
    "--full-subdir",
    type=str,
    default="full",
    help="Subdirectory name holding full releases beneath data-root (default: full)",
)
args = parser.parse_args()

console = Console()

# %%[markdown]
# # Subject Overlap Across Releases
#
# This script inspects the mini releases to answer:
# - Are subject IDs reused across releases?
# - Which tasks contribute subjects to each release?
# - How many subjects would be excluded by the shared exclusion list?

# %%
TASKS = [
    "RestingState",
    "surroundSupp",
    "contrastChangeDetection",
    "seqLearning6target",
    "seqLearning8target",
    "symbolSearch",
    "DespicableMe",
    "DiaryOfAWimpyKid",
    "FunwithFractals",
    "ThePresent",
]
RELEASES = [f"R{i}" for i in range(1, 12)]
CACHE_DIR = args.data_root

# %%
subject_to_release: defaultdict[str, set[str]] = defaultdict(set)
release_task_subjects: dict[tuple[str, str], set[str]] = {}

for release in RELEASES:
    for task in TASKS:
        try:
            ds = load_challenge_dataset(
                task=task,
                release=release,
                variant=args.dataset_variant,
                cache_dir=CACHE_DIR,
                full_subdir=args.full_subdir,
                description_fields=["subject"],
            )
        except Exception:
            continue

        subjects = set()
        for base_ds in ds.datasets:
            desc = base_ds.description
            subj = desc.get("subject", "") if hasattr(desc, "get") else ""
            if subj:
                subjects.add(subj)
                subject_to_release[subj].add(release)
        if subjects:
            release_task_subjects[(release, task)] = subjects

# %%
# Build summary table
summary = Table(
    title=f"Subjects per Release ({args.dataset_variant})",
    box=box.MINIMAL_DOUBLE_HEAD,
)
summary.add_column("Release", justify="left", style="bold cyan")
summary.add_column("# Subjects")
summary.add_column("Overlap with other releases")
summary.add_column("Excluded")

for release in RELEASES:
    subjects_in_release = {
        subj
        for (rel, _), subs in release_task_subjects.items()
        if rel == release
        for subj in subs
    }
    if not subjects_in_release:
        continue
    overlap = sorted(
        subj for subj in subjects_in_release if len(subject_to_release[subj]) > 1
    )
    excluded = sorted(subjects_in_release & set(SUBJECTS_TO_REMOVE))
    summary.add_row(
        release,
        str(len(subjects_in_release)),
        str(len(overlap)),
        str(len(excluded)),
    )

console.print(summary)

# %%[markdown]
# ## Subjects appearing in multiple releases

# %%
overlap_table = Table(
    title=f"Subjects spanning multiple releases ({args.dataset_variant})", box=box.SIMPLE
)
overlap_table.add_column("Subject", style="magenta")
overlap_table.add_column("Releases")

for subj, releases in sorted(subject_to_release.items()):
    if len(releases) <= 1:
        continue
    overlap_table.add_row(subj, ", ".join(sorted(releases)))

if len(overlap_table.rows) == 0:
    console.print("[green]No subject appears in more than one release.[/green]")
else:
    console.print(overlap_table)

# %%[markdown]
# ## Task coverage per release

# %%
coverage = Table(
    title=f"Task coverage ({args.dataset_variant})", box=box.SIMPLE_HEAD
)
coverage.add_column("Release", style="bold cyan")
coverage.add_column("Tasks with data")

for release in RELEASES:
    tasks_present = [
        task for (rel, task), subs in release_task_subjects.items() if rel == release
    ]
    if not tasks_present:
        continue
    coverage.add_row(release, ", ".join(sorted(tasks_present)))

console.print(coverage)

# %%[markdown]
# ## Notes
#
# Run this script with `uv run python notebooks/004_subject_overlap.py`. The output will summarise
# subject uniqueness, which releases contain excluded IDs, and which tasks contribute to each release.
# Adjust `CACHE_DIR` if your data root differs from `./data`.
