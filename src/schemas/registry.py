"""Lookup of output schemas by (dataset, task), checked against configs/data_prep.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.schemas.base import OutputSchema
from src.schemas.tasks import ALL_SCHEMAS

TASKS: dict[tuple[str, str], type[OutputSchema]] = {(s.dataset, s.task): s for s in ALL_SCHEMAS}


def get_schema(dataset: str, task: str) -> type[OutputSchema]:
    """Output schema for one task."""
    try:
        return TASKS[(dataset, task)]
    except KeyError:
        raise KeyError(f"no schema for {dataset}/{task}; known: {sorted(TASKS)}") from None


def check_against_config(config_path: str | Path) -> None:
    """Fail if any schema's type, column or class order differs from the data config."""
    tasks = yaml.safe_load(Path(config_path).read_text())["tasks"]
    config_keys = {(ds, t) for ds, ts in tasks.items() for t in ts}
    errors = [f"{k}: in config but has no schema" for k in sorted(config_keys - TASKS.keys())]
    errors += [f"{k}: has a schema but not in config" for k in sorted(TASKS.keys() - config_keys)]

    for (ds, t), schema in TASKS.items():
        spec = tasks.get(ds, {}).get(t)
        if spec is None:
            continue
        if spec["type"] != schema.task_type:
            errors.append(f"{ds}/{t}: type {spec['type']} != {schema.task_type}")
        if spec["column"] != schema.column:
            errors.append(f"{ds}/{t}: column {spec['column']} != {schema.column}")
        # Binary tasks have no class list; their single field is the target column
        expected = [str(c) for c in spec.get("classes", [spec["column"]])]
        if expected != schema.labels():
            errors.append(f"{ds}/{t}: classes {expected} != {schema.labels()}")
    if errors:
        raise ValueError("schemas out of sync with config:\n" + "\n".join(errors))
