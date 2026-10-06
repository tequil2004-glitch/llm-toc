"""Reproducibility helpers for multi-seed MARL training."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, device: Optional[torch.device] = None) -> None:
    """Seed Python, NumPy, and PyTorch RNGs deterministically.

    Args:
        seed: Integer seed.
        device: Torch device. CUDA is seeded with ``torch.cuda.manual_seed_all``
            when CUDA is available.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Best-effort determinism. cuDNN benchmarking is disabled to favor
    # reproducibility over throughput.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
