"""Load processed splits with the filters each evaluation setting needs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.schemas.base import OutputSchema


def load_split(
    data_root: Path,
    dataset: str,
    split: str,
    schema: type[OutputSchema],
    all_cmsb: bool = False,
    limit: int | None = None,
) -> tuple[pd.DataFrame, Path, list[str]]:
    """Return (rows to run on, parquet path, applied filters); ids are cast to str."""
    if split == "test":
        raise ValueError("test is read only in notebooks/99_final_eval.ipynb")
    path = data_root / "processed" / split / f"{dataset}_{split}.parquet"
    df = pd.read_parquet(path)
    filters = []
    # Main CMSB setting is annotated-only: unannotated 'other' posts are a source shortcut
    if dataset == "cmsb" and not all_cmsb:
        df = df[df["annotated"]]
        filters.append("annotated only")
    # TODO: EDOS B/C scope is open (data/CLAUDE.md); for now the gold-sexist subset only
    if schema.task_type == "multiclass":
        df = df[df[schema.column].notna()]
        filters.append("gold-sexist only")
    if limit is not None:
        df = df.head(limit)
        filters.append(f"first {limit} rows")
    df = df.reset_index(drop=True).assign(id=lambda d: d["id"].astype(str))
    return df, path, filters
