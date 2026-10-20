"""Reproducibility helpers: seeds, pinned revisions, file hashes and environment info."""

from __future__ import annotations

import hashlib
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


def get_hf_token() -> str | None:
    """HF_TOKEN from the env, else Colab secrets (only reachable inside the kernel)."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        from google.colab import userdata

        return userdata.get("HF_TOKEN")
    except Exception:
        return None


def resolve_revision(repo_id: str, revision: str, token: str | None) -> str:
    """Pin a branch or tag to its commit sha so the run records the exact weights."""
    from huggingface_hub import HfApi

    return HfApi().model_info(repo_id, revision=revision, token=token).sha


def set_seeds(seed: int) -> None:
    """Seed Python, NumPy, PyTorch (CPU and CUDA) and transformers."""
    import torch
    import transformers

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment_info(project_root: Path) -> dict[str, Any]:
    """Package versions, GPU and git state, saved next to every run."""
    import torch
    import transformers

    def run(cmd: list[str]) -> str:
        try:
            return subprocess.run(
                cmd, capture_output=True, text=True, check=True, cwd=project_root
            ).stdout.strip()
        except Exception as e:
            return f"unavailable: {e}"

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": run(["git", "rev-parse", "HEAD"]),
        "git_dirty": run(["git", "status", "--porcelain"]) != "",
        "pip_freeze": run([sys.executable, "-m", "pip", "freeze"]).splitlines(),
    }
