"""Seed and CUDA guards used by every executable workload."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False


def require_authorized_cuda(expected_gpu_substring: str | None = None) -> str:
    """Reject local/CPU execution and return the verified physical GPU name."""

    if not os.environ.get("KUBERNETES_SERVICE_HOST"):
        raise RuntimeError("scientific workloads must run inside a Kubernetes pod")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the Kubernetes pod")
    gpu_name = torch.cuda.get_device_name(0)
    allowed = ("H200", "H100", "RTX 8000", "Quadro RTX 8000")
    if not any(item.lower() in gpu_name.lower() for item in allowed):
        raise RuntimeError(f"unauthorized GPU: {gpu_name}")
    if expected_gpu_substring and expected_gpu_substring.lower() not in gpu_name.lower():
        raise RuntimeError(
            f"actual GPU {gpu_name!r} does not match expected {expected_gpu_substring!r}"
        )
    return gpu_name
