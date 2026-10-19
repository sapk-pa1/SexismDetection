"""One OutputSchema per (dataset, task); class order must match configs/data_prep.yaml."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field

from src.schemas.base import OutputSchema, TaskType


class CmsbBinary(OutputSchema):
    dataset: ClassVar[str] = "cmsb"
    task: ClassVar[str] = "binary"
    task_type: ClassVar[TaskType] = "binary"
    column: ClassVar[str] = "sexist"

    sexist: bool


class EdosTaskA(OutputSchema):
    dataset: ClassVar[str] = "edos"
    task: ClassVar[str] = "task_a"
    task_type: ClassVar[TaskType] = "binary"
    column: ClassVar[str] = "sexist"

    sexist: bool


class EdosTaskB(OutputSchema):
    dataset: ClassVar[str] = "edos"
    task: ClassVar[str] = "task_b"
    task_type: ClassVar[TaskType] = "multiclass"
    column: ClassVar[str] = "category"

    category: Literal["threats", "derogation", "animosity", "prejudiced_discussions"]


class EdosTaskC(OutputSchema):
    dataset: ClassVar[str] = "edos"
    task: ClassVar[str] = "task_c"
    task_type: ClassVar[TaskType] = "multiclass"
    column: ClassVar[str] = "vector"

    # Official vector codes; names live in the prompt template
    vector: Literal[
        "1.1", "1.2", "2.1", "2.2", "2.3", "3.1", "3.2", "3.3", "3.4", "4.1", "4.2"
    ]


class ExistTask11(OutputSchema):
    dataset: ClassVar[str] = "exist"
    task: ClassVar[str] = "task_1_1"
    task_type: ClassVar[TaskType] = "binary"
    column: ClassVar[str] = "sexist"

    sexist: bool


class ExistTask13(OutputSchema):
    dataset: ClassVar[str] = "exist"
    task: ClassVar[str] = "task_1_3"
    task_type: ClassVar[TaskType] = "multilabel"
    column: ClassVar[str] = "labels"

    # Aliases are the official EXIST names, so JSON keys match the gold files
    ideological_inequality: bool = Field(alias="IDEOLOGICAL-INEQUALITY")
    stereotyping_dominance: bool = Field(alias="STEREOTYPING-DOMINANCE")
    objectification: bool = Field(alias="OBJECTIFICATION")
    sexual_violence: bool = Field(alias="SEXUAL-VIOLENCE")
    misogyny_non_sexual_violence: bool = Field(alias="MISOGYNY-NON-SEXUAL-VIOLENCE")


ALL_SCHEMAS: tuple[type[OutputSchema], ...] = (
    CmsbBinary,
    EdosTaskA,
    EdosTaskB,
    EdosTaskC,
    ExistTask11,
    ExistTask13,
)
