"""JSON-constrained generation from an OutputSchema (outlines, xgrammar backend).

Fallback only: the grammar forces a schema-valid answer, so a refusal can't show up in
the output. Log-prob scoring with its Yes/No mass stays the primary method.
"""

from __future__ import annotations

from typing import Any

import torch
from outlines import from_transformers
from outlines.backends import get_json_schema_logits_processor
from pydantic import ValidationError
from transformers import LogitsProcessorList

from src.prompts.builder import ChatOptions, PromptTemplate, render
from src.schemas.base import OutputSchema


class JsonGenerator:
    """Greedy JSON generation constrained to one schema; build once per (model, schema)."""

    def __init__(self, model: Any, tokenizer: Any, schema: type[OutputSchema]):
        self.model, self.tokenizer, self.schema = model, tokenizer, schema
        # The outlines wrapper is only used to compile the grammar; generation is plain HF
        self.processor = get_json_schema_logits_processor(
            "xgrammar", from_transformers(model, tokenizer), schema.json_schema()
        )

    @torch.inference_mode()
    def __call__(
        self,
        template: PromptTemplate,
        text: str,
        max_new_tokens: int = 256,
        enable_thinking: bool = False,
        chat: ChatOptions | None = None,
    ) -> tuple[str, OutputSchema | None, str | None]:
        """Return (raw output, parsed schema or None, parse error or None)."""
        # The grammar starts after the prefill, so the prefill must not contain JSON
        prompt = render(self.tokenizer, template.json_messages(text), template.prefill, enable_thinking, chat)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        self.processor.reset()
        out = self.model.generate(
            **inputs,
            logits_processor=LogitsProcessorList([self.processor]),
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        raw = self.tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        try:
            return raw, self.schema.model_validate_json(raw), None
        except ValidationError as e:
            # Truncation at max_new_tokens is the usual cause; stored, never dropped
            return raw, None, str(e)
