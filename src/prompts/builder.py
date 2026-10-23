"""Load per-task prompt templates and render them into chat prompts."""

from __future__ import annotations

import hashlib
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.prompts.fewshot import Shot
from src.schemas.base import OutputSchema

TEMPLATE_DIR = Path(__file__).parent / "templates"
# sha256 of every frozen prompt variant (v2, T1-T4, X); a variant missing here or with another hash is refused
VARIANT_LOCK = Path(__file__).resolve().parents[2] / "configs" / "prompt_variants.lock"
OPTION_LETTERS = string.ascii_uppercase

Messages = list[dict[str, str]]


@dataclass(frozen=True)
class ChatOptions:
    """Per-model prompt options from configs/llms.yaml, applied on top of the task template."""

    prefill: str = ""  # added after the template prefill, e.g. "<score>" for Granite Guardian
    chat_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptTemplate:
    """One task's prompt template; `sha256` of the YAML file goes into every prediction."""

    prompt_id: str
    version: int
    source: str
    system: str
    yes_no: str | None  # binary and multilabel
    options: str | None  # multiclass
    json: str  # generation backend
    prefill: str
    labels: dict[str, dict[str, str]]  # label -> {name, definition}, in schema order
    sha256: str
    variant: str = ""  # "" = the frozen v1 file; else e.g. "v2", "T3" (file {task}.{variant}.yaml)
    logprob_prefill: str = ""  # reply prefix for log-prob scoring, e.g. "Answer: " (v2 only)

    @classmethod
    def load(
        cls,
        schema: type[OutputSchema],
        template_dir: Path = TEMPLATE_DIR,
        variant: str | None = None,
        lock: Path = VARIANT_LOCK,
    ) -> PromptTemplate:
        """Load the template for a schema and check it covers exactly its labels; variants must match the lock."""
        name = f"{schema.prompt_id()}.{variant}.yaml" if variant else f"{schema.prompt_id()}.yaml"
        path = template_dir / name
        raw = path.read_bytes()
        if variant:
            # Variants are frozen before any run (docs/12 §3.4-3.6); an edited file must not run silently
            frozen = yaml.safe_load(lock.read_text()) if lock.exists() else {}
            sha = hashlib.sha256(raw).hexdigest()
            if (frozen or {}).get(name) != sha:
                raise ValueError(f"{name} is not frozen in {lock.name} (or was edited after freezing)")
        spec = yaml.safe_load(raw)
        # Keys are quoted in YAML but str() guards against bare numbers like 1.1
        labels = {str(k): v for k, v in spec["labels"].items()}
        if list(labels) != schema.labels():
            raise ValueError(f"{path}: labels {list(labels)} != schema {schema.labels()}")
        needed = "options" if schema.task_type == "multiclass" else "yes_no"
        if not spec.get(needed):
            raise ValueError(f"{path}: {schema.task_type} task needs a '{needed}' template")
        if schema.task_type == "multiclass" and len(labels) > len(OPTION_LETTERS):
            raise ValueError(f"{path}: more classes than option letters")
        return cls(
            prompt_id=f"{schema.prompt_id()}.{variant}" if variant else schema.prompt_id(),
            version=spec["version"],
            source=spec["source"],
            system=spec["system"],
            yes_no=spec.get("yes_no"),
            options=spec.get("options"),
            json=spec["json"],
            prefill=spec.get("prefill") or "",
            labels=labels,
            sha256=hashlib.sha256(raw).hexdigest(),
            variant=variant or "",
            logprob_prefill=spec.get("logprob_prefill") or "",
        )

    def _messages(self, user: str, shots: list[tuple[str, str]] = ()) -> Messages:
        # Few-shot: each shot is a prior user/assistant turn, so the reply still starts with the answer
        turns = [
            m for q, a in shots for m in ({"role": "user", "content": q}, {"role": "assistant", "content": a})
        ]
        return [{"role": "system", "content": self.system}, *turns, {"role": "user", "content": user}]

    def yes_no_messages(self, text: str, label: str, shots: list[Shot] = ()) -> Messages:
        """One Yes/No question about one label, after any shots for that label."""
        lab = self.labels[label]

        def ask(t: str) -> str:
            return self.yes_no.format(name=lab["name"], definition=lab["definition"], text=t)

        return self._messages(ask(text), [(ask(s.text), s.answer) for s in shots])

    def option_letters(self) -> list[str]:
        """Answer letters, one per class in schema order."""
        return list(OPTION_LETTERS[: len(self.labels)])

    def option_order(self, perm: int = 0) -> list[str]:
        """Classes in display order: schema order rotated left by `perm` (letters stay A, B, ...)."""
        classes = list(self.labels)
        if not 0 <= perm < len(classes):
            raise ValueError(f"option perm {perm} outside 0..{len(classes) - 1}")
        return classes[perm:] + classes[:perm]

    def option_messages(self, text: str, shots: list[Shot] = (), perm: int = 0) -> Messages:
        """One question listing every class as a lettered option; shot answers become letters."""
        letters, order = self.option_letters(), self.option_order(perm)
        options = "\n".join(
            f"{letter}) {self.labels[c]['name']}: {self.labels[c]['definition']}"
            for letter, c in zip(letters, order, strict=True)
        )
        letter_of = dict(zip(order, letters, strict=True))
        turns = [(self.options.format(options=options, text=s.text), letter_of[s.answer]) for s in shots]
        return self._messages(self.options.format(options=options, text=text), turns)

    def json_messages(self, text: str) -> Messages:
        """One request for the whole OutputSchema as JSON."""
        fields = "\n".join(f'- "{k}" ({v["name"]}): {v["definition"]}' for k, v in self.labels.items())
        return self._messages(self.json.format(fields=fields, text=text))


def render(
    tokenizer: Any,
    messages: Messages,
    prefill: str = "",
    enable_thinking: bool = False,
    chat: ChatOptions | None = None,
) -> str:
    """Apply the chat template; the prompt ends where the model's reply starts."""
    chat = chat or ChatOptions()
    # enable_thinking is read by Qwen/SmolLM3/Nemotron templates and ignored by the others
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        **chat.chat_kwargs,
    )
    return prompt + prefill + chat.prefill
