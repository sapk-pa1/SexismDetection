"""Next-token log-probability scoring: Yes/No per label, or one lettered option per class."""

from __future__ import annotations

import hashlib
from typing import Any

import torch

from src.prompts.builder import ChatOptions, PromptTemplate, render
from src.prompts.fewshot import Shot, Shots
from src.schemas.base import OutputSchema
from src.schemas.records import LabelScore

# Answer tokens summed into P(Yes) / P(No); variants that aren't one token are dropped
YES_VARIANTS = ("Yes", " Yes", "yes", " yes", "YES")
NO_VARIANTS = ("No", " No", "no", " no", "NO")


def answer_token_ids(tokenizer: Any, variants: tuple[str, ...]) -> list[int]:
    """Unique token ids of the variants that encode to exactly one token."""
    ids = []
    for v in variants:
        enc = tokenizer.encode(v, add_special_tokens=False)
        if len(enc) == 1 and enc[0] not in ids:
            ids.append(enc[0])
    if not ids:
        raise ValueError(f"no single-token answer among {variants}")
    return ids


@torch.inference_mode()
def next_token_probs(model: Any, tokenizer: Any, prompt: str) -> torch.Tensor:
    """Softmax over the vocabulary for the token after `prompt`."""
    # The chat template already adds BOS, so don't add special tokens twice
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    logits = model(**inputs, logits_to_keep=1).logits[0, -1].float()
    return torch.softmax(logits, dim=-1)


def _top_k(tokenizer: Any, probs: torch.Tensor, k: int) -> list[tuple[str, float]]:
    top = torch.topk(probs, k)
    return [(tokenizer.decode([i]), p.item()) for p, i in zip(top.values, top.indices, strict=True)]


def score_yes_no(
    model: Any,
    tokenizer: Any,
    template: PromptTemplate,
    text: str,
    label: str,
    top_k: int = 10,
    enable_thinking: bool = False,
    chat: ChatOptions | None = None,
    shots: list[Shot] = (),
) -> LabelScore:
    """Score one label from the next-token Yes/No distribution."""
    # template.logprob_prefill is "" for the frozen v1 prompts, so their rendered text is unchanged
    prompt = render(
        tokenizer,
        template.yes_no_messages(text, label, shots),
        template.logprob_prefill,
        enable_thinking=enable_thinking,
        chat=chat,
    )
    probs = next_token_probs(model, tokenizer, prompt)
    p_yes = probs[answer_token_ids(tokenizer, YES_VARIANTS)].sum().item()
    p_no = probs[answer_token_ids(tokenizer, NO_VARIANTS)].sum().item()
    mass = p_yes + p_no
    score = p_yes / mass if mass > 0 else None
    return LabelScore(
        label=label,
        probs={"Yes": p_yes, "No": p_no},
        mass=mass,
        score=score,
        pred=None if score is None else score >= 0.5,  # dev-tuned thresholds are applied in 99 (docs/12 §3.1)
        top_k=_top_k(tokenizer, probs, top_k),
        tie=mass > 0 and p_yes == p_no,
        rendered_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
    )


def score_options(
    model: Any,
    tokenizer: Any,
    template: PromptTemplate,
    text: str,
    column: str,
    top_k: int = 10,
    enable_thinking: bool = False,
    chat: ChatOptions | None = None,
    shots: list[Shot] = (),
    perm: int = 0,
) -> LabelScore:
    """Score a single-label task from the next-token distribution over option letters.

    `perm` rotates which class sits behind letter A (option-order control); probs stay keyed by class.
    """
    prompt = render(
        tokenizer,
        template.option_messages(text, shots, perm),
        template.logprob_prefill,
        enable_thinking=enable_thinking,
        chat=chat,
    )
    probs = next_token_probs(model, tokenizer, prompt)
    class_probs = {
        cls: probs[answer_token_ids(tokenizer, (letter, f" {letter}"))].sum().item()
        for letter, cls in zip(template.option_letters(), template.option_order(perm), strict=True)
    }
    mass = sum(class_probs.values())
    # max() keeps the first of exactly tied classes, i.e. the one shown earlier; `tie` records that it happened
    best = max(class_probs, key=class_probs.get)
    top2 = sorted(class_probs.values(), reverse=True)[:2]
    return LabelScore(
        label=column,
        probs=class_probs,
        mass=mass,
        score=class_probs[best] / mass if mass > 0 else None,
        pred=best if mass > 0 else None,
        top_k=_top_k(tokenizer, probs, top_k),
        tie=mass > 0 and len(top2) == 2 and top2[0] == top2[1],
        rendered_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
    )


def score_example(
    model: Any,
    tokenizer: Any,
    schema: type[OutputSchema],
    template: PromptTemplate,
    text: str,
    top_k: int = 10,
    enable_thinking: bool = False,
    chat: ChatOptions | None = None,
    shots: Shots | None = None,
    perm: int = 0,
) -> list[LabelScore]:
    """All scores for one post: one per label (binary, multilabel) or one (multiclass)."""
    shots = shots or {}
    if schema.task_type == "multiclass":
        return [
            score_options(
                model,
                tokenizer,
                template,
                text,
                schema.column,
                top_k,
                enable_thinking,
                chat,
                shots.get(schema.column, ()),
                perm,
            )
        ]
    if perm:
        raise ValueError("option permutations apply to multiclass tasks only")
    return [
        score_yes_no(
            model, tokenizer, template, text, label, top_k, enable_thinking, chat, shots.get(label, ())
        )
        for label in schema.labels()
    ]
