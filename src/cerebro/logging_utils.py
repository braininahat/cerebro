"""Shared Rich console/progress helpers used across Cerebro scripts."""

from __future__ import annotations

from functools import lru_cache

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


@lru_cache(maxsize=1)
def get_console() -> Console:
    """Return a shared console instance with consistent styling."""
    return Console()


def create_progress(*, transient: bool = False) -> Progress:
    """Create a Rich progress bar with Cerebro defaults."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=get_console(),
        transient=transient,
    )


__all__ = ["create_progress", "get_console"]
