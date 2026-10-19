"""Input and output records shared by LLM and encoder runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class Example(BaseModel):
    """One post to classify."""

    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    dataset: str
    split: Literal["train", "dev", "test"]

    @classmethod
    def from_row(cls, row: Mapping[str, Any], dataset: str) -> Example:
        """Build from a processed-parquet row; models see `text_norm`, not raw text."""
        return cls(id=str(row["id"]), text=row["text_norm"], dataset=dataset, split=row["split"])


class LabelScore(BaseModel):
    """Next-token scores for one (example, label) pair, or one multiclass prompt."""

    label: str  # label name (binary/multilabel) or target column (multiclass)
    probs: dict[str, float]  # raw next-token P per answer: Yes/No or class name
    mass: float  # sum of probs; low mass marks a soft refusal
    score: float | None  # P(Yes)/mass, or the argmax class share; None when mass is 0
    pred: bool | str | None  # None when mass is 0
    top_k: list[tuple[str, float]]  # logged so refusals can be inspected
    refused: bool | None = None  # None until the dev mass threshold is frozen


class Prediction(BaseModel):
    """One JSONL line in predictions.jsonl, keyed by example id."""

    id: str
    dataset: str
    task: str
    model_id: str
    revision: str
    backend: Literal["logprob", "generate"]
    prompt_id: str
    prompt_sha256: str
    scores: list[LabelScore] = []
    raw_output: str | None = None  # generation backend: always logged, parsed or not
    parsed: dict[str, Any] | None = None  # OutputSchema dumped by alias
    parse_error: str | None = None
