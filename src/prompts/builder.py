"""Load per-task prompt templates and render them into chat prompts."""

from __future__ import annotations

import hashlib
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.schemas.base import OutputSchema

TEMPLATE_DIR = Path(__file__).parent / "templates"
OPTION_LETTERS = string.ascii_uppercase

Messages = list[dict[str, str]]


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

    @classmethod
    def load(cls, schema: type[OutputSchema], template_dir: Path = TEMPLATE_DIR) -> PromptTemplate:
        """Load the template for a schema and check it covers exactly its labels."""
        path = template_dir / f"{schema.prompt_id()}.yaml"
        raw = path.read_bytes()
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
            prompt_id=schema.prompt_id(),
            version=spec["version"],
            source=spec["source"],
            system=spec["system"],
            yes_no=spec.get("yes_no"),
            options=spec.get("options"),
            json=spec["json"],
            prefill=spec.get("prefill") or "",
            labels=labels,
            sha256=hashlib.sha256(raw).hexdigest(),
        )

    def _messages(self, user: str) -> Messages:
        return [{"role": "system", "content": self.system}, {"role": "user", "content": user}]

    def yes_no_messages(self, text: str, label: str) -> Messages:
        """One Yes/No question about one label."""
        lab = self.labels[label]
        return self._messages(self.yes_no.format(name=lab["name"], definition=lab["definition"], text=text))

    def option_letters(self) -> list[str]:
        """Answer letters, one per class in schema order."""
        return list(OPTION_LETTERS[: len(self.labels)])

    def option_messages(self, text: str) -> Messages:
        """One question listing every class as a lettered option."""
        options = "\n".join(
            f"{letter}) {lab['name']}: {lab['definition']}"
            for letter, lab in zip(self.option_letters(), self.labels.values(), strict=True)
        )
        return self._messages(self.options.format(options=options, text=text))

    def json_messages(self, text: str) -> Messages:
        """One request for the whole OutputSchema as JSON."""
        fields = "\n".join(f'- "{k}" ({v["name"]}): {v["definition"]}' for k, v in self.labels.items())
        return self._messages(self.json.format(fields=fields, text=text))


def render(tokenizer: Any, messages: Messages, prefill: str = "", enable_thinking: bool = False) -> str:
    """Apply the chat template; the prompt ends where the model's reply starts."""
    # enable_thinking is read by Qwen templates and ignored by the others
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
    return prompt + prefill
