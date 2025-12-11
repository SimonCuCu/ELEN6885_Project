"""Random seed utilities for reproducibility.

This module provides functions to set random seeds for all libraries
to ensure reproducible training and evaluation.
"""

import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility.

    Sets seeds for Python's random, NumPy, and PyTorch (CPU and CUDA).

    Args:
        seed: Random seed value (default: 42)

    Example:
        >>> set_seed(42)
        >>> # Now all random operations will be reproducible
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may slow down training)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> torch.device:
    """Get the appropriate torch device.

    Args:
        device: Device specification. Options:
            - "auto": Automatically select GPU if available
            - "cuda": Use GPU (raises error if not available)
            - "cpu": Use CPU
            - None: Same as "auto"

    Returns:
        torch.device object

    Example:
        >>> device = get_device("auto")
        >>> model = model.to(device)
    """
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    else:
        return torch.device(device)


def worker_init_fn(worker_id: int) -> None:
    """Initialize worker random seed for DataLoader.

    Use this as worker_init_fn in DataLoader to ensure reproducibility
    with multiple workers.

    Args:
        worker_id: Worker ID provided by DataLoader

    Example:
        >>> loader = DataLoader(
        ...     dataset,
        ...     num_workers=4,
        ...     worker_init_fn=worker_init_fn
        ... )
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

