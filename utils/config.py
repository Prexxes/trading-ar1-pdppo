"""Configuration helpers used by training scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TrainingPaths:
    """Resolved paths for logs and checkpoints."""

    root: Path
    checkpoint_dir: Path
    log_csv: Path
