"""Single-example LLM inference: the entry point notebooks call per example ID."""

from __future__ import annotations

from typing import Any, Literal

from src.model.generate import JsonGenerator
from src.model.logprob import score_example
from src.prompts.builder import ChatOptions, PromptTemplate
from src.schemas.base import OutputSchema
from src.schemas.records import Example, Prediction


def infer_single(
    model: Any,
    tokenizer: Any,
    example: Example,
    schema: type[OutputSchema],
    *,
    model_id: str,
    revision: str,
    backend: Literal["logprob", "generate"] = "logprob",
    template: PromptTemplate | None = None,
    generator: JsonGenerator | None = None,
    mass_threshold: float | None = None,
    top_k: int = 10,
    max_new_tokens: int = 256,
    enable_thinking: bool = False,
    chat: ChatOptions | None = None,
) -> Prediction:
    """Run one post through one task and return a JSONL-ready Prediction.

    Args:
        mass_threshold: soft-refusal cutoff on Yes/No (or option) mass. Leave None until
            it is frozen on dev; `refused` then stays None.
        generator: reuse a JsonGenerator across calls; one is built if None.
    """
    if example.dataset != schema.dataset:
        raise ValueError(f"example from {example.dataset}, schema for {schema.dataset}")
    template = template or PromptTemplate.load(schema)
    pred = Prediction(
        id=example.id,
        dataset=schema.dataset,
        task=schema.task,
        model_id=model_id,
        revision=revision,
        backend=backend,
        prompt_id=template.prompt_id,
        prompt_sha256=template.sha256,
    )

    if backend == "logprob":
        pred.scores = score_example(
            model, tokenizer, schema, template, example.text, top_k, enable_thinking, chat
        )
        if mass_threshold is not None:
            for s in pred.scores:
                s.refused = s.mass < mass_threshold
        # A refused label is its own outcome, so no parsed answer if any label refused
        if all(s.pred is not None and not s.refused for s in pred.scores):
            if schema.task_type == "multilabel":
                value = {s.label for s in pred.scores if s.pred}
            else:
                value = pred.scores[0].pred
            pred.parsed = schema.from_prediction(value).model_dump(by_alias=True)
        return pred

    generator = generator or JsonGenerator(model, tokenizer, schema)
    raw, parsed, error = generator(template, example.text, max_new_tokens, enable_thinking, chat)
    pred.raw_output = raw
    pred.parsed = parsed.model_dump(by_alias=True) if parsed is not None else None
    pred.parse_error = error
    return pred
