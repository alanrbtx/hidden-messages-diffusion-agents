"""Stable checksums for tensors and artifact files."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    """Hash logical tensor content without mutating the source tensor."""

    contiguous = tensor.detach().to(device="cpu").contiguous()
    payload = contiguous.view(torch.uint8).numpy().tobytes()
    metadata = f"{tuple(contiguous.shape)}|{contiguous.dtype}".encode()
    return hashlib.sha256(metadata + b"\0" + payload).hexdigest()
