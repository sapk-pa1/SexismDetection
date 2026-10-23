"""Seeded, stratified few-shot sets from train: select, freeze to YAML (IDs only), load."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.schemas.base import OutputSchema
from src.utils.repro import sha256_file

FEWSHOT_DIR = Path(__file__).resolve().parents[2] / "configs" / "fewshot"


@dataclass(frozen=True)
class Shot:
    """One demonstration: a train post and the reply it gets ("Yes"/"No" or a class name)."""

    id: str
    answer: str
    text: str = ""  # filled from the train parquet by load_shots; never stored in git


Shots = dict[str, list[Shot]]  # label (binary/multilabel) or target column (multiclass) -> shots


def shots_path(dataset: str, task: str, k: int, seed: int, root: Path = FEWSHOT_DIR) -> Path:
    """Frozen shot file for one task, e.g. configs/fewshot/edos/task_b_k1_s42.yaml."""
    return root / dataset / f"{task}_k{k}_s{seed}.yaml"


def _eligible(df: pd.DataFrame, schema: type[OutputSchema], max_words: int) -> tuple[pd.DataFrame, list[str]]:
    """Train rows allowed as shots, and the filters applied (stored in the frozen file)."""
    # Leak/dup rows could put a shot's text in dev/test; short posts keep prompts short
    keep = df["leak_group"].isna() & df["dup_group"].isna() & ~df["empty_norm"]
    keep &= df[schema.column].notna() & (df["text_norm"].str.split().str.len() <= max_words)
    filters = [
        "no leak_group",
        "no dup_group",
        "not empty_norm",
        "hard label present",
        f"<= {max_words} words",
    ]
    if schema.dataset == "cmsb":
        keep &= df["annotated"] & df["of_id"].isna()
        filters.append("annotated originals (no of_id)")
    if schema.dataset == "exist":
        keep &= df["lang"] == "en"
        filters.append("lang en")
    # Sort by id so the seeded draw doesn't depend on parquet row order
    return df[keep].sort_values("id"), filters


def select_shots(
    train_df: pd.DataFrame, schema: type[OutputSchema], k: int = 1, seed: int = 42, max_words: int = 50
) -> tuple[Shots, list[str]]:
    """k shots per class (multiclass), per answer (binary), or k Yes + k No per label (multilabel)."""
    pool, filters = _eligible(train_df, schema, max_words)
    rng = np.random.default_rng(seed)

    def draw(rows: pd.DataFrame, answer: str) -> list[Shot]:
        if len(rows) < k:
            raise ValueError(f"{schema.prompt_id()}: only {len(rows)} eligible rows for answer {answer!r}")
        return [Shot(str(i), answer) for i in rows.sample(n=k, random_state=rng)["id"]]

    col = pool[schema.column]
    if schema.task_type == "binary":
        groups = {schema.labels()[0]: draw(pool[col == 1], "Yes") + draw(pool[col == 0], "No")}
    elif schema.task_type == "multiclass":
        groups = {schema.column: [s for cls in schema.labels() for s in draw(pool[col == cls], cls)]}
    else:
        # No shots are sexist posts without the label, so they teach the category boundary
        has = {lab: col.map(lambda v, lab=lab: lab in v) for lab in schema.labels()}
        sexist = col.map(len) > 0
        groups = {
            lab: draw(pool[has[lab]], "Yes") + draw(pool[sexist & ~has[lab]], "No") for lab in schema.labels()
        }
    # Seeded shuffle so the last shot (recency bias) isn't always the same answer
    return {key: [shots[i] for i in rng.permutation(len(shots))] for key, shots in groups.items()}, filters


def freeze_shots(
    data_root: Path,
    schema: type[OutputSchema],
    k: int = 1,
    seed: int = 42,
    max_words: int = 50,
    root: Path = FEWSHOT_DIR,
) -> Path:
    """Select shots from train and write them (IDs only) to their frozen file; never overwrites."""
    path = shots_path(schema.dataset, schema.task, k, seed, root)
    if path.exists():
        raise FileExistsError(f"{path} is frozen; use a new seed or k instead of overwriting")
    train_file = data_root / "processed" / "train" / f"{schema.dataset}_train.parquet"
    shots, filters = select_shots(pd.read_parquet(train_file), schema, k, seed, max_words)
    spec = {
        "dataset": schema.dataset,
        "task": schema.task,
        "k": k,
        "seed": seed,
        "max_words": max_words,
        "created": date.today().isoformat(),
        "train_file": str(train_file.relative_to(data_root)),
        "train_sha256": sha256_file(train_file),
        "filters": filters,
        "shots": {key: [{"id": s.id, "answer": s.answer} for s in group] for key, group in shots.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))
    return path


def load_shots(path: Path, data_root: Path) -> Shots:
    """Read a frozen shot file and fill in each post's text_norm from the train parquet it names."""
    spec = yaml.safe_load(path.read_text())
    train_file = data_root / spec["train_file"]
    if sha256_file(train_file) != spec["train_sha256"]:
        raise ValueError(f"{train_file} changed since {path} was frozen")
    texts = pd.read_parquet(train_file, columns=["id", "text_norm"]).set_index("id")["text_norm"]
    return {
        str(key): [Shot(str(s["id"]), str(s["answer"]), texts[str(s["id"])]) for s in group]
        for key, group in spec["shots"].items()
    }
