"""Load Hugging Face instruct models at a pinned revision."""

from __future__ import annotations

from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    FineGrainedFP8Config,
)


def load_hf_model(
    repo_id: str,
    revision: str,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str | dict = "auto",
    token: str | None = None,
    fp8_dequantize: bool = False,
    trust_remote_code: bool = False,
) -> tuple[Any, Any]:
    """Load an instruct model and tokenizer at a pinned revision."""
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id, revision=revision, token=token, trust_remote_code=trust_remote_code
    )
    kwargs = dict(
        revision=revision,
        dtype=dtype,
        device_map=device_map,
        token=token,
        trust_remote_code=trust_remote_code,
    )
    if fp8_dequantize:
        # FP8-only checkpoints (Ministral 3): A100 has no native FP8, so dequantize to `dtype` at load
        kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(repo_id, **kwargs)
    except ValueError:
        # Multimodal checkpoints (e.g. Ministral 3, model_type mistral3) have no causal-LM mapping
        model = AutoModelForImageTextToText.from_pretrained(repo_id, **kwargs)
    model.eval()
    return model, tokenizer
