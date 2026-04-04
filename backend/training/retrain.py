"""Auto retrain when enough deliveries have been completed since last training run."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from backend.db import queries

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_COMPLETION_THRESHOLD = 50


def check_and_maybe_retrain() -> None:
    """If completions since last retrain >= 50, run train_models and stamp checkpoint."""
    try:
        n = queries.count_completions_since_retrain()
        if n < _COMPLETION_THRESHOLD:
            return
        subprocess.run(
            [sys.executable, "-m", "backend.training.train_models"],
            cwd=str(PROJECT_ROOT),
            check=True,
        )
        queries.mark_retrain_done_now()
    except Exception:
        log.exception("auto-retrain skipped or failed")
