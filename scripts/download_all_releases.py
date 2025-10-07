"""Utility to cache EEG2025 mini and full releases across tasks.

Run with the desired cache root via `EEG2025_DATA_DIR`. By default the script
downloads every task for releases R1–R11 for both the mini (Braindecode-ready)
subset and the full BIDS exports. Use CLI flags to narrow tasks or releases if
needed.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from eegdash.dataset import EEGChallengeDataset

from cerebro.data import download_all_raws
from cerebro.logging_utils import create_progress, get_console

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"^\s*\[EEGChallengeDataset\] EEG 2025 Competition Data Notice",
    module="eegdash",
)

console = get_console()

ALL_RELEASES: tuple[str, ...] = tuple(f"R{i}" for i in range(1, 12))
ALL_TASKS: tuple[str, ...] = (
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
)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--releases",
        nargs="+",
        default=list(ALL_RELEASES),
        help="Releases to cache (default: all R1–R11)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(ALL_TASKS),
        help="Tasks to cache (default: all supported tasks)",
    )
    parser.add_argument(
        "--skip-mini",
        action="store_true",
        help="Skip mini releases (EEGLAB .set cache)",
    )
    parser.add_argument(
        "--skip-full",
        action="store_true",
        help="Skip full BIDS releases",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override cache directory (default: EEG2025_DATA_DIR or ./data)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of download jobs to run in parallel (default: 1)",
    )
    parser.add_argument(
        "--dataset-workers",
        type=int,
        default=-1,
        help=(
            "Worker count passed to EEGChallengeDataset (default: -1 for all cores). "
            "Set to 1 if you hit I/O limits."
        ),
    )
    parser.add_argument(
        "--materialize-raw",
        action="store_true",
        help="Force-load raw files for each dataset to avoid lazy downloads at training time.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _normalise(values: Sequence[str], allowed: Sequence[str], label: str) -> list[str]:
    normalised = []
    allowed_index = {v.lower(): idx for idx, v in enumerate(allowed)}
    allowed_lookup = {v.lower(): v for v in allowed}
    for value in values:
        key = value.lower()
        if key not in allowed_lookup:
            raise ValueError(
                f"Unknown {label}: {value}. Allowed values: {sorted(allowed)}"
            )
        normalised.append(allowed_lookup[key])
    return sorted(set(normalised), key=lambda v: allowed_index[v.lower()])


@dataclass(frozen=True)
class DownloadJob:
    release: str
    task: str
    mini: bool
    cache_dir: Path

    def label(self) -> str:
        prefix = "mini" if self.mini else "full"
        return f"{prefix}:{self.release}:{self.task}"


def cache_release(
    task: str,
    release: str,
    *,
    mini: bool,
    cache_dir: Path,
    description_fields: Sequence[str] | None = None,
    dataset_workers: int = -1,
    materialize_raw: bool = False,
) -> tuple[bool, str | None]:
    try:
        dataset = EEGChallengeDataset(
            task=task,
            release=release,
            mini=mini,
            cache_dir=cache_dir,
            description_fields=list(description_fields)
            if description_fields
            else None,
            n_jobs=dataset_workers,
        )
        if materialize_raw:
            download_all_raws(dataset, n_jobs=dataset_workers)
    except AssertionError as exc:
        message = str(exc)
        if "datasets should not be an empty iterable" in message:
            return False, "no recordings"
        raise
    return True, None


def _build_jobs(
    *,
    releases: Sequence[str],
    tasks: Sequence[str],
    data_root: Path,
    description_fields: Sequence[str],
    skip_mini: bool,
    skip_full: bool,
) -> tuple[list[DownloadJob], Path | None]:
    jobs: list[DownloadJob] = []
    if not skip_mini:
        for release in releases:
            for task in tasks:
                jobs.append(
                    DownloadJob(
                        release=release,
                        task=task,
                        mini=True,
                        cache_dir=data_root,
                    )
                )

    full_root: Path | None = None
    if not skip_full:
        full_root = (data_root / "full").resolve()
        full_root.mkdir(parents=True, exist_ok=True)
        for release in releases:
            for task in tasks:
                jobs.append(
                    DownloadJob(
                        release=release,
                        task=task,
                        mini=False,
                        cache_dir=full_root,
                    )
                )

    return jobs, full_root


def _run_jobs(
    jobs: Sequence[DownloadJob],
    *,
    description_fields: Sequence[str],
    dataset_workers: int,
    workers: int,
    materialize_raw: bool,
) -> list[tuple[DownloadJob, bool, str | None]]:
    results: list[tuple[DownloadJob, bool, str | None]] = []
    total = len(jobs)
    if total == 0:
        return results

    workers = max(1, workers)

    def _submit(job: DownloadJob) -> tuple[bool, str | None]:
        return cache_release(
            task=job.task,
            release=job.release,
            mini=job.mini,
            cache_dir=job.cache_dir,
            description_fields=description_fields,
            dataset_workers=dataset_workers,
            materialize_raw=materialize_raw,
        )

    if workers == 1:
        with create_progress() as progress:
            task_id = progress.add_task("Caching releases", total=total)
            for job in jobs:
                success, reason = _submit(job)
                status = (
                    "[green]OK[/green]"
                    if success
                    else f"[yellow]SKIP ({reason})[/yellow]"
                )
                progress.advance(task_id)
                console.log(f"{job.label()} :: {status}")
                results.append((job, success, reason))
    else:
        console.print(f"Running up to [bold]{workers}[/bold] jobs in parallel…")
        with create_progress() as progress:
            task_id = progress.add_task("Caching releases", total=total)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_job = {
                    executor.submit(_submit, job): job for job in jobs
                }
                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        success, reason = future.result()
                    except Exception as exc:
                        console.log(f"[red]{job.label()} :: ERROR ({exc})[/red]")
                        raise
                    status = (
                        "[green]OK[/green]"
                        if success
                        else f"[yellow]SKIP ({reason})[/yellow]"
                    )
                    progress.advance(task_id)
                    console.log(f"{job.label()} :: {status}")
                    results.append((job, success, reason))

    return results


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)

    data_root = args.data_root or Path(os.environ.get("EEG2025_DATA_DIR", "data"))
    data_root = data_root.expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    releases = _normalise(args.releases, ALL_RELEASES, label="release")
    tasks = _normalise(args.tasks, ALL_TASKS, label="task")

    if args.skip_full and args.skip_mini:
        raise ValueError("Both mini and full releases are skipped; nothing to cache.")

    console.rule("EEG2025 Cache Sync")
    console.print(f"Caching to [bold]{data_root}[/bold]")
    console.print(f"Releases: [cyan]{', '.join(releases)}[/cyan]")
    console.print(f"Tasks: [cyan]{', '.join(tasks)}[/cyan]")

    description_fields = ["subject", "task", "run", "release"]

    jobs, full_root = _build_jobs(
        releases=releases,
        tasks=tasks,
        data_root=data_root,
        description_fields=description_fields,
        skip_mini=args.skip_mini,
        skip_full=args.skip_full,
    )

    if not args.skip_mini:
        console.print("\n[bold]Mini releases (EEGLAB .set)[/bold]")
    if not args.skip_full:
        console.print("[bold]Full releases (BIDS)[/bold]")

    if jobs:
        console.print(f"\nQueued [bold]{len(jobs)}[/bold] download jobs.")

    results = _run_jobs(
        jobs,
        description_fields=description_fields,
        dataset_workers=args.dataset_workers,
        workers=args.workers,
        materialize_raw=args.materialize_raw,
    )

    downloaded = [job.label() for job, success, _ in results if success]
    skipped = [job.label() for job, success, reason in results if not success and reason]

    console.print("\n[bold green]All requested releases processed.[/bold green]")
    if downloaded:
        console.print(f"  Cached [green]{len(downloaded)}[/green] entries.")
    if skipped:
        console.print(
            f"  Skipped [yellow]{len(skipped)}[/yellow] entries with no recordings:"
        )
        for entry in skipped:
            console.print(f"    - {entry}")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
