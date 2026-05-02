"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: logging_setup.py.
Description:
    Logging factory for dual rich-console + file output.
    Provides a reusable setup_logging function that configures a file
    handler with timestamps and a Rich console handler.

Dependencies:
    - Python >= 3.10.
    - rich.
"""

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(script_name, tag="", log_dir=None, console=None):
    """Configure logging with a rich console handler and a file handler.

    Clears any pre-existing handlers so the function can be called more
    than once in the same process.

    Args:
        script_name (str): Base name used for the log file.
        tag (str): Optional suffix appended to the log file name.
        log_dir (Path | None): Directory for the log file. When None,
            only the console handler is created (no file output).
        console (Console | None): Rich console instance to use for the
            console handler. A new one is created when None.

    Returns:
        tuple[logging.Logger, Console]: (logger, rich console).
    """
    if console is None:
        console = Console()

    # Clear existing handlers to allow re-initialisation.
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # File handler with timestamps for complete logging.
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{tag}" if tag else ""
        log_file = log_dir / f"{script_name}{suffix}.log"
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        file_handler.setLevel(logging.INFO)
        root.addHandler(file_handler)

    # Rich console handler for formatted interactive output.
    rich_handler = RichHandler(
        console=console,
        show_time=False,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    rich_handler.setLevel(logging.INFO)
    root.addHandler(rich_handler)

    logger = logging.getLogger(script_name)
    return logger, console
