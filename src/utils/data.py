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
    """Return (rows to run on, parquet path, applied filters); ids are cast to str.

    On test the frame is label-blind: only id, text_norm, split (and CMSB `annotated`) are read.
    `annotated` is source metadata, but every unannotated CMSB post is non-sexist.
    """
    path = data_root / "processed" / split / f"{dataset}_{split}.parquet"
    blind = split == "test"
    # Test labels are read only in 99_final_eval, so never load label columns here
    columns = ["id", "text_norm", "split"] + (["annotated"] if dataset == "cmsb" else [])
    df = pd.read_parquet(path, columns=columns if blind else None)
    filters = ["label-blind"] if blind else []
    # Main CMSB setting is annotated-only: unannotated 'other' posts are a source shortcut
    if dataset == "cmsb" and not all_cmsb:
        df = df[df["annotated"]]
        filters.append("annotated only")
    # TODO: EDOS B/C scope is open (data/CLAUDE.md); gold-sexist subset on dev/train, all posts on test.
    # EXIST 1.2 runs on all tweets (NO is a class); metrics score its hard-majority subset
    if dataset == "edos" and schema.task_type == "multiclass" and not blind:
        df = df[df[schema.column].notna()]
        filters.append("gold-sexist only")
    if limit is not None:
        df = df.head(limit)
        filters.append(f"first {limit} rows")
    df = df.reset_index(drop=True).assign(id=lambda d: d["id"].astype(str))
    return df, path, filters
