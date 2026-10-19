"""Base class for per-task LLM output schemas."""

from __future__ import annotations

import json
from typing import Any, ClassVar, Literal, get_args

from pydantic import BaseModel, ConfigDict

TaskType = Literal["binary", "multiclass", "multilabel"]


class OutputSchema(BaseModel):
    """What a model outputs for one task; subclasses set the ClassVars and fields."""

    # extra="forbid" makes the JSON schema reject unknown keys in constrained decoding
    model_config = ConfigDict(extra="forbid", populate_by_name=True, frozen=True)

    dataset: ClassVar[str]
    task: ClassVar[str]
    task_type: ClassVar[TaskType]
    column: ClassVar[str]  # target column in the processed parquet

    @classmethod
    def prompt_id(cls) -> str:
        """Template path under src/prompts/templates, without the suffix."""
        return f"{cls.dataset}/{cls.task}"

    @classmethod
    def labels(cls) -> list[str]:
        """Labels in config class order: classes (multiclass) or field aliases."""
        if cls.task_type == "multiclass":
            (field,) = cls.model_fields.values()
            return list(get_args(field.annotation))
        return [f.alias or name for name, f in cls.model_fields.items()]

    @classmethod
    def json_schema(cls) -> str:
        """JSON schema (by alias) used for constrained decoding."""
        return json.dumps(cls.model_json_schema(by_alias=True))

    @classmethod
    def from_prediction(cls, pred: Any) -> OutputSchema:
        """Build from a bool (binary), a class (multiclass) or a set of labels."""
        if cls.task_type == "binary":
            return cls(**{cls.labels()[0]: bool(pred)})
        if cls.task_type == "multiclass":
            return cls(**{cls.column: pred})
        return cls.model_validate({lab: lab in pred for lab in cls.labels()})

    def value(self) -> bool | str | list[str]:
        """Prediction in gold-file form: bool, class name or list of active labels."""
        dumped = self.model_dump(by_alias=True)
        if self.task_type == "multilabel":
            return [lab for lab in self.labels() if dumped[lab]]
        return next(iter(dumped.values()))
