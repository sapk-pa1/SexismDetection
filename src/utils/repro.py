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


def _run(cmd: list[str], cwd: Path) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=cwd).stdout.strip()
    except Exception as e:
        return f"unavailable: {e}"


def git_state(project_root: Path) -> dict[str, Any]:
    """Commit and dirty flag of the repo; no torch import, so usable before HF_HOME is set."""
    return {
        "git_commit": _run(["git", "rev-parse", "HEAD"], project_root),
        "git_dirty": _run(["git", "status", "--porcelain"], project_root) != "",
    }


def environment_info(project_root: Path) -> dict[str, Any]:
    """Package versions, GPU and git state, saved next to every run."""
    import torch
    import transformers

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        **git_state(project_root),
        "pip_freeze": _run([sys.executable, "-m", "pip", "freeze"], project_root).splitlines(),
    }
