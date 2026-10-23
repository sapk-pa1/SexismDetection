"""Registry of prompted LLMs (configs/llms.yaml): repo, pinned revision and per-model options."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LLMSpec:
    """One model entry; extra options are logged in each run's config."""

    name: str
    repo_id: str
    revision: str
    params: str
    licence: str
    gated: bool
    thinking: str
    vram_gb: float  # scheduling estimate for parallel jobs, see configs/llms.yaml
    prefill: str = ""
    chat_kwargs: dict[str, Any] = field(default_factory=dict)
    fp8_dequantize: bool = False
    trust_remote_code: bool = False
    blocked: str = ""  # non-empty: run_inference refuses the model, with this reason
    display_name: str = ""  # name used in reports and figures; falls back to the short name


def load_llm_registry(path: Path) -> dict[str, LLMSpec]:
    """All models by short name; unknown keys fail so typos can't silently drop an option."""
    models = yaml.safe_load(path.read_text())["models"]
    return {name: LLMSpec(name=name, **spec) for name, spec in models.items()}
