"""
Logging utilities for TribeBot T9.
Configures structured console + file logging and optional W&B integration.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logging(
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    filename: str = "tribebot_t9.log",
) -> logging.Logger:
    """
    Configure root logger to write to both stdout and a rotating file.
    Returns the root logger so callers can use it directly.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / filename

    fmt = "[%(asctime)s] %(levelname)s %(name)s — %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, encoding="utf-8"),
    ]
    for h in handlers:
        h.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(log_level)
    # Avoid duplicate handlers on re-import
    if not root.handlers:
        for h in handlers:
            root.addHandler(h)

    return root


class MetricsLogger:
    """
    Lightweight in-process metrics store with optional Weights & Biases upload.
    Falls back gracefully when wandb is not installed.
    """

    def __init__(self, project: Optional[str] = None, run_name: Optional[str] = None) -> None:
        self._history: Dict[str, list] = {}
        self._wandb_run = None

        if project:
            try:
                import wandb  # type: ignore
                self._wandb_run = wandb.init(project=project, name=run_name)
            except ImportError:
                logging.getLogger(__name__).warning(
                    "wandb not installed — skipping W&B logging. "
                    "Install with: pip install wandb"
                )

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        for k, v in metrics.items():
            self._history.setdefault(k, []).append(v)

        if self._wandb_run is not None:
            self._wandb_run.log(metrics, step=step)

    def get_history(self, key: str) -> list:
        return self._history.get(key, [])

    def finish(self) -> None:
        if self._wandb_run is not None:
            self._wandb_run.finish()
