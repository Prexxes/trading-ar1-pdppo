"""Configuration helpers used by training scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TrainingPaths:
    """Resolved paths for logs and checkpoints.

    Attributes:
        root: Root directory for the training run.
        checkpoint_dir: Directory containing model checkpoints.
        log_csv: CSV file used for metric logging.
    """

    root: Path
    checkpoint_dir: Path
    log_csv: Path
