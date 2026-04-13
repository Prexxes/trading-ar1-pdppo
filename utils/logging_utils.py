"""Utilities for CSV logging and directory management."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        The same path after ensuring it exists.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_csv_row(path: Path, fieldnames: Iterable[str], row: Mapping[str, object]) -> None:
    """Append a row to a CSV file and write a header when needed.

    Args:
        path: Target CSV file path.
        fieldnames: Ordered CSV column names.
        row: Mapping containing the row values to append.
    """
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
