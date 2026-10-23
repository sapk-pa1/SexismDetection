"""Single-example LLM inference: the entry point notebooks call per example ID."""

from __future__ import annotations

from typing import Any, Literal

from src.model.generate import JsonGenerator
from src.model.logprob import score_example
from src.model.text import generate_text, parse_option, parse_yes_no
from src.prompts.builder import ChatOptions, PromptTemplate, render
from src.prompts.fewshot import Shots
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
    backend: Literal["logprob", "generate", "text"] = "logprob",
    template: PromptTemplate | None = None,
    generator: JsonGenerator | None = None,
    mass_threshold: float | None = None,
    top_k: int = 10,
    max_new_tokens: int = 256,
    enable_thinking: bool = False,
    chat: ChatOptions | None = None,
    shots: Shots | None = None,
    option_perm: int = 0,
) -> Prediction:
    """Run one post through one task and return a JSONL-ready Prediction.

    Args:
        mass_threshold: soft-refusal cutoff on Yes/No (or option) mass. Leave None until
            it is frozen on dev; `refused` then stays None.
        generator: reuse a JsonGenerator across calls; one is built if None.
        shots: frozen few-shot set (log-prob backend only); None is zero-shot.
        option_perm: rotation of the lettered options (multiclass, log-prob and text backends).
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
            model, tokenizer, schema, template, example.text, top_k, enable_thinking, chat, shots, option_perm
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

    if shots:
        raise ValueError("few-shot prompts are implemented for the logprob backend only")
    if backend == "text":
        return _text_reply(
            pred, model, tokenizer, schema, template, example.text, max_new_tokens, chat, option_perm
        )
    generator = generator or JsonGenerator(model, tokenizer, schema)
    raw, parsed, error = generator(template, example.text, max_new_tokens, enable_thinking, chat)
    pred.raw_output = raw
    pred.parsed = parsed.model_dump(by_alias=True) if parsed is not None else None
    pred.parse_error = error
    return pred


def _text_reply(
    pred: Prediction,
    model: Any,
    tokenizer: Any,
    schema: type[OutputSchema],
    template: PromptTemplate,
    text: str,
    max_new_tokens: int,
    chat: ChatOptions | None,
    option_perm: int,
) -> Prediction:
    """Free-text reply to the same prompt the log-prob scorer sees, read with the fixed parser (docs/12 §3.3)."""
    if schema.task_type == "multiclass":
        messages = template.option_messages(text, perm=option_perm)
    elif schema.task_type == "binary":
        messages = template.yes_no_messages(text, schema.labels()[0])
    else:
        raise ValueError("the text backend covers binary and multiclass tasks only")
    raw = generate_text(model, tokenizer, render(tokenizer, messages, chat=chat), max_new_tokens)
    pred.raw_output = raw
    if schema.task_type == "multiclass":
        order = template.option_order(option_perm)
        letters = dict(zip(template.option_letters(), order, strict=True))
        value, rule = parse_option(raw, letters, {c: template.labels[c]["name"] for c in order})
    else:
        value, _ = parse_yes_no(raw)
        rule = "yes_no" if value is not None else "unparseable"
    pred.parse_rule = rule
    if value is None:
        pred.parse_error = "unparseable reply"
    else:
        pred.parsed = schema.from_prediction(value).model_dump(by_alias=True)
    return pred
