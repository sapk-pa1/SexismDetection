"""Free-text replies for the format check and self-explanations, read with the fixed parser of docs/12 §3.3."""

from __future__ import annotations

import re
from typing import Any

MARKUP = re.compile(r"[*#`]")
# Rule 1: the reply opens with a letter that is clearly an option label, so "A post that..." is not option A
LEADING_LETTER = re.compile(r"^\s*\(?([A-K])\)?(?:[).:\]]|\s*$|\s*\n)")
# Rule 2: "answer is C", "option: C", "category - C"
KEYWORD_LETTER = re.compile(r"\b(?:answer|option|category)\b\s*(?:is|:|-)?\s*\(?([A-K])\b", re.IGNORECASE)
YES_NO = re.compile(r"^\W*(yes|no)\b[\s.,:;!-]*(.*)", re.IGNORECASE | re.DOTALL)


def generate_text(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> str:
    """Greedy reply to a rendered prompt, special tokens removed."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def parse_option(
    reply: str, letter_to_class: dict[str, str], class_names: dict[str, str]
) -> tuple[str | None, str]:
    """(class or None, rule) from a free-text reply to a lettered-option prompt.

    Args:
        letter_to_class: letters as shown in this prompt (after any option rotation) -> class.
        class_names: class -> display name used in the prompt, for rule 3.
    """
    text = MARKUP.sub("", reply)
    m = LEADING_LETTER.match(text)
    if m and m.group(1) in letter_to_class:
        return letter_to_class[m.group(1)], "leading_letter"
    m = KEYWORD_LETTER.search(text)
    if m and m.group(1).upper() in letter_to_class:
        return letter_to_class[m.group(1).upper()], "keyword_letter"
    low = text.lower()
    # Whole words only: EXIST names such as "direct" must not match "directly"
    hits = [c for c, name in class_names.items() if re.search(rf"\b{re.escape(name.lower())}\b", low)]
    if len(hits) == 1:
        return hits[0], "class_name"
    return None, "unparseable"


def parse_yes_no(reply: str) -> tuple[bool | None, str]:
    """(answer or None, rationale) from a reply that should start with Yes or No."""
    m = YES_NO.match(MARKUP.sub("", reply))
    if not m:
        return None, reply.strip()
    return m.group(1).lower() == "yes", m.group(2).strip()
