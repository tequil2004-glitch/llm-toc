"""Unified CSV + TensorBoard logger.

Writes scalar metrics to:

1. A shared CSV file (``metrics.csv``) with columns
   ``[run_name, step, key, value]`` for easy multi-seed aggregation and
   shaded-curve plotting.
2. A per-run TensorBoard event file under ``tb/<run_name>`` for live
   monitoring.

TensorBoard is optional: when ``tensorboard`` is not installed, the
logger silently degrades to CSV-only.
"""

from __future__ import annotations

import csv
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


class Logger:
    """Thread-safe CSV + TensorBoard scalar logger.

    The CSV file is shared across all runs (one row per
    ``(run_name, step, key, value)`` tuple) so that multi-seed results
    can be aggregated trivially with pandas. TensorBoard event files
    are written under ``tensorboard_dir/<run_name>``.

    Attributes:
        run_name: Logical run name (e.g. ``llm_toc_mappo_seed42``).
        csv_path: Path to the shared CSV file.
        tensorboard_dir: Directory containing per-run TB event files.
    """

    _csv_lock = threading.Lock()
    _file_handles: Dict[str, Any] = {}

    def __init__(
        self,
        csv_path: Union[str, Path],
        tensorboard_dir: Union[str, Path],
        run_name: str,
    ) -> None:
        """Initialize the logger.

        Args:
            csv_path: Path to the shared CSV file. Created with headers
                if it does not exist.
            tensorboard_dir: Directory for per-run TensorBoard logs.
            run_name: Logical run name.
        """
        self.csv_path = Path(csv_path)
        self.tensorboard_dir = Path(tensorboard_dir)
        self.run_name = run_name
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)

        self._tb_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._tb_writer = SummaryWriter(
                log_dir=str(self.tensorboard_dir / run_name))
        except Exception:  # noqa: BLE001
            self._tb_writer = None

        self._buffered: List[Dict[str, Any]] = []
        self._init_csv()

    # ------------------------------------------------------------------
    def _init_csv(self) -> None:
        """Create the CSV file with headers if it does not exist."""
        with self._csv_lock:
            if not self.csv_path.exists():
                with open(self.csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["run_name", "step", "key", "value"])

    # ------------------------------------------------------------------
    def log_scalar(self, key: str, value: float, step: int) -> None:
        """Buffer a scalar for both CSV and TensorBoard.

        Args:
            key: Metric key (e.g. ``"train/success_rate"``).
            value: Scalar value.
            step: Global step counter.
        """
        try:
            v = float(value)
        except (TypeError, ValueError):
            v = float("nan")
        self._buffered.append({"run_name": self.run_name, "step": int(step),
                                "key": key, "value": v})
        if self._tb_writer is not None:
            try:
                self._tb_writer.add_scalar(key, v, int(step))
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    def log_scalars(self, prefix: str, mapping: Dict[str, float],
                    step: int) -> None:
        """Log a dict of scalars under a shared prefix."""
        for k, v in mapping.items():
            self.log_scalar(f"{prefix}/{k}", v, step)

    # ------------------------------------------------------------------
    def flush(self) -> None:
        """Flush buffered rows to the CSV file and the TB writer."""
        with self._csv_lock:
            if self._buffered:
                with open(self.csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    for row in self._buffered:
                        writer.writerow([row["run_name"], row["step"],
                                          row["key"], row["value"]])
                self._buffered.clear()
        if self._tb_writer is not None:
            try:
                self._tb_writer.flush()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Flush and close all open handles."""
        self.flush()
        if self._tb_writer is not None:
            try:
                self._tb_writer.close()
            except Exception:  # noqa: BLE001
                pass
            self._tb_writer = None
