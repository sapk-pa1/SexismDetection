"""Data models: per-task output schemas and the records written by every run."""

from src.schemas.base import OutputSchema, TaskType
from src.schemas.records import Example, LabelScore, Prediction
from src.schemas.registry import TASKS, check_against_config, get_schema
from src.schemas.tasks import (
    ALL_SCHEMAS,
    CmsbBinary,
    EdosTaskA,
    EdosTaskB,
    EdosTaskC,
    ExistTask11,
    ExistTask13,
)

__all__ = [
    "ALL_SCHEMAS",
    "TASKS",
    "CmsbBinary",
    "EdosTaskA",
    "EdosTaskB",
    "EdosTaskC",
    "Example",
    "ExistTask11",
    "ExistTask13",
    "LabelScore",
    "OutputSchema",
    "Prediction",
    "TaskType",
    "check_against_config",
    "get_schema",
]
