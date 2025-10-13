"""Rich logging utilities for console and file output.

Extracted from notebook 04_train_challenge1.py to provide consistent
logging across CLI and notebook environments.
"""

import logging
import re
from pathlib import Path
from rich.console import Console
from rich.logging import RichHandler


class PlainFormatter(logging.Formatter):
    """Formatter that strips Rich markup tags for clean file output."""

    def format(self, record):
        # Create copy to avoid mutating original record
        record = logging.makeLogRecord(record.__dict__)
        # Strip Rich markup tags: [bold], [/bold], [green], etc.
        record.msg = re.sub(r'\[/?[^\]]+\]', '', str(record.msg))
        return super().format(record)


def setup_logging(
    log_file: Path,
    logger_name: str = "cerebro",
    log_level: int = logging.INFO,
    log_format: str = "%(message)s",
    log_datefmt: str = "[%X]",
    log_file_mode: str = "w",
    rich_tracebacks: bool = True,
    rich_markup: bool = True,
) -> logging.Logger:
    """Setup Rich logging with console and file handlers.

    Args:
        log_file: Path to log file (will be created if doesn't exist)
        logger_name: Name of logger to configure
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_format: Format string for log messages
        log_datefmt: Date format for timestamps
        log_file_mode: File mode ('w' = overwrite, 'a' = append)
        rich_tracebacks: Enable rich traceback formatting
        rich_markup: Enable rich markup in console output

    Returns:
        Configured logger instance

    Example:
        >>> log_file = Path("outputs/run_20250113/train.log")
        >>> logger = setup_logging(log_file)
        >>> logger.info("[bold green]Training started![/bold green]")
    """
    # Ensure log file directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Configure Rich handler for console
    rich_handler = RichHandler(
        rich_tracebacks=rich_tracebacks,
        markup=rich_markup
    )

    # Configure file handler with plain formatting
    file_handler = logging.FileHandler(log_file, mode=log_file_mode)
    file_handler.setFormatter(PlainFormatter(log_format))

    # Configure root logger explicitly to capture all loggers (including Lightning)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear any existing handlers (prevents duplicates if running multiple times)
    root_logger.handlers.clear()

    # Add our handlers
    root_logger.addHandler(rich_handler)
    root_logger.addHandler(file_handler)

    # Get named logger
    logger = logging.getLogger(logger_name)

    return logger


def get_console() -> Console:
    """Get Rich console instance for standalone use.

    Returns:
        Console instance for printing formatted output

    Example:
        >>> console = get_console()
        >>> console.print("[bold]Hello[/bold] [green]world[/green]")
    """
    return Console()
