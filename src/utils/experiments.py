"""Experiment folders: one dated folder per batch holding its runs, a manifest, logs and a heartbeat."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.repro import git_state, sha256_file
from src.utils.run_io import load_predictions

EXPERIMENTS_DIR = "experiments"
STAMP_FMT = "%Y%m%d-%H%M%S"
NAME_RE = re.compile(r"[a-z0-9][a-z0-9-]*")
FOLDER_RE = re.compile(r"(\d{8}-\d{6})_experiment_([a-z0-9][a-z0-9-]*)")
MANIFEST_COLUMNS = [
    "run_name", "kind", "model", "model_id", "revision", "dataset", "task", "split", "method", "backend",
    "prompt_version", "template_variant", "option_perm", "shots", "seed", "sample", "n_examples", "n_pred",
    "status", "output_file", "output_sha256", "finished",
]  # fmt: skip


def _jsonable(obj: Any) -> Any:
    # Round-trip through JSON so tuples and lists compare equal after a reload
    return json.loads(json.dumps(obj, default=str))


def _write_atomic(path: Path, text: str) -> None:
    # Write then rename, so a disconnect never leaves a half-written file on Drive
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def read_experiment(exp_dir: Path) -> dict[str, Any]:
    return json.loads((exp_dir / "experiment.json").read_text())


def runs_dir(exp_dir: Path) -> Path:
    """Folder holding the experiment's run folders (`runs/`, or the old location for registered batches)."""
    return (exp_dir / read_experiment(exp_dir)["runs_dir"]).resolve()


def experiment_date(exp_dir: Path) -> str:
    """YYYYMMDD of the experiment's creation, used as the `{date}` part of its run names."""
    return read_experiment(exp_dir)["created"][:8]


def find_experiment(output_root: Path, name: str) -> Path | None:
    """The one existing folder `{stamp}_experiment_{name}` under OUTPUT_ROOT/experiments, or None."""
    root = output_root / EXPERIMENTS_DIR
    hits = [p for p in root.glob(f"*_experiment_{name}") if p.is_dir() and FOLDER_RE.fullmatch(p.name)]
    if len(hits) > 1:
        raise ValueError(f"{len(hits)} folders for experiment {name!r}: {sorted(p.name for p in hits)}")
    return hits[0] if hits else None


def open_experiment(
    output_root: Path,
    name: str,
    config: dict[str, Any],
    *,
    purpose: str,
    protocol_ref: str,
    project_root: Path,
    sample: bool = False,
    ignore: tuple[str, ...] = (),
) -> Path:
    """Create `{YYYYMMDD-HHMMSS}_experiment_{name}` (`{name}-sample` in sample mode), or reopen it if the config matches.

    Reopening with a changed config is refused (keys in `ignore`, e.g. scheduling options, may differ):
    a changed setting needs a new experiment name, so finished runs are never mixed with new ones.
    """
    if not NAME_RE.fullmatch(name) or name.endswith("-sample"):
        raise ValueError(
            f"experiment name {name!r}: use lowercase letters, digits and '-' (no '-sample' suffix)"
        )
    full = f"{name}-sample" if sample else name
    config = _jsonable(config)
    existing = find_experiment(output_root, full)
    if existing is not None:
        old = read_experiment(existing)["config"]
        diff = sorted(k for k in (old.keys() | config.keys()) - set(ignore) if old.get(k) != config.get(k))
        if diff:
            raise ValueError(
                f"{existing.name} exists with different settings {diff}; use a new experiment name"
            )
        return existing
    stamp = datetime.now().strftime(STAMP_FMT)
    exp_dir = output_root / EXPERIMENTS_DIR / f"{stamp}_experiment_{full}"
    (exp_dir / "runs").mkdir(parents=True)
    (exp_dir / "logs").mkdir()
    meta = {
        "experiment": exp_dir.name,
        "name": full,
        "created": stamp,
        "mode": "sample" if sample else "full",
        "purpose": purpose,
        "protocol_ref": protocol_ref,
        **git_state(project_root),
        "runs_dir": "runs",
        "config": config,
    }
    _write_atomic(exp_dir / "experiment.json", json.dumps(meta, indent=2))
    return exp_dir


def register_experiment(
    output_root: Path,
    name: str,
    runs_path: Path,
    *,
    created: str,
    purpose: str,
    protocol_ref: str,
    config: dict[str, Any],
) -> Path:
    """Describe an existing batch of run folders as an experiment, without moving or changing any run file."""
    if not NAME_RE.fullmatch(name):
        raise ValueError(f"experiment name {name!r}: use lowercase letters, digits and '-'")
    datetime.strptime(created, STAMP_FMT)
    exp_dir = output_root / EXPERIMENTS_DIR / f"{created}_experiment_{name}"
    if exp_dir.exists():
        raise FileExistsError(f"{exp_dir.name} is already registered")
    exp_dir.mkdir(parents=True)
    meta = {
        "experiment": exp_dir.name,
        "name": name,
        "created": created,
        "mode": "full",
        "purpose": purpose,
        "protocol_ref": protocol_ref,
        "registered": datetime.now().isoformat(timespec="seconds"),
        # Relative, so the registration survives moving OUTPUT_ROOT between Drive and a local copy
        "runs_dir": os.path.relpath(runs_path.resolve(), exp_dir.resolve()),
        "config": _jsonable(config),
    }
    _write_atomic(exp_dir / "experiment.json", json.dumps(meta, indent=2))
    write_manifest(exp_dir)
    return exp_dir


def build_manifest(exp_dir: Path) -> pd.DataFrame:
    """One row per run folder, read from the folders themselves (never edited by hand)."""
    rows = []
    for cfg_path in sorted(runs_dir(exp_dir).glob("*/config.json")):
        cfg = json.loads(cfg_path.read_text())
        # Batch folders and other non-run folders have no dataset/split; they are not runs
        if not {"run_name", "dataset", "split"} <= cfg.keys():
            continue
        run = cfg_path.parent
        output = run / cfg.get("output_file", "predictions.jsonl")
        complete = run / "complete.json"
        done = complete.exists()
        rows.append({
            "run_name": cfg["run_name"],
            "kind": cfg.get("kind", "llm"),
            "model": cfg.get("model_name") or cfg.get("model"),
            "model_id": cfg.get("model_id"),
            "revision": cfg.get("revision"),
            "dataset": cfg["dataset"],
            "task": cfg.get("task"),
            "split": cfg["split"],
            "method": cfg.get("method"),
            "backend": cfg.get("backend"),
            "prompt_version": cfg.get("prompt_version") or "v1",
            "template_variant": cfg.get("template_variant") or "T0",
            "option_perm": cfg.get("option_perm"),
            "shots": cfg.get("shots", 0),
            "seed": cfg.get("seed"),
            "sample": cfg.get("sample") is not None,
            "n_examples": cfg.get("n_examples"),
            "n_pred": sum(1 for _ in output.open()) if output.exists() else 0,
            "status": "complete" if done else "partial",
            "output_file": output.name,
            "output_sha256": sha256_file(output) if done and output.exists() else "",
            "finished": json.loads(complete.read_text()).get("finished", "") if done else "",
        })  # fmt: skip
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def write_manifest(exp_dir: Path) -> pd.DataFrame:
    """Rebuild manifest.csv and README.md from the run folders; returns the manifest."""
    man = build_manifest(exp_dir)
    _write_atomic(exp_dir / "manifest.csv", man.to_csv(index=False))
    meta = read_experiment(exp_dir)
    counts = man.status.value_counts().to_dict() if len(man) else {}
    lines = [
        f"# {meta['experiment']}",
        "",
        f"- **Purpose:** {meta['purpose']}",
        f"- **Protocol:** {meta['protocol_ref']}",
        f"- **Mode:** {meta['mode']}" + (" (sanity check only; never used for evaluation)" if meta["mode"] == "sample" else ""),
        f"- **Created:** {meta['created']}; git `{str(meta.get('git_commit', 'n/a'))[:10]}`",
        f"- **Runs:** {len(man)} ({counts}); one row each in `manifest.csv`, rebuilt from the run folders.",
        "- Config in `experiment.json`; job logs in `logs/`; live progress in `heartbeat.json`.",
    ]  # fmt: skip
    _write_atomic(exp_dir / "README.md", "\n".join(lines) + "\n")
    return man


def write_heartbeat(exp_dir: Path, state: dict[str, Any]) -> None:
    """Progress snapshot for checking a background batch from Drive without opening the notebook."""
    payload = {"updated": datetime.now().isoformat(timespec="seconds"), **_jsonable(state)}
    _write_atomic(exp_dir / "heartbeat.json", json.dumps(payload, indent=2))


def load_experiment_runs(
    exp_dirs: list[Path], split: str, kind: str = "llm", allow_sample: bool = False
) -> list[dict[str, Any]]:
    """Finished non-sample runs of these experiments, each checked against its manifest row (for 99).

    Refuses sample experiments, sample runs, finished runs missing from the manifest and outputs whose
    sha256 differs from the manifest. `allow_sample` is only for 99's own sanity mode, whose outputs go
    to separate sample folders and never into results/.
    """
    runs = []
    for exp in exp_dirs:
        meta = read_experiment(exp)
        if meta["mode"] != "full" and not allow_sample:
            raise ValueError(
                f"{exp.name} is a sample experiment; it is a sanity check, never an evaluation input"
            )
        man = pd.read_csv(exp / "manifest.csv", dtype=str, keep_default_na=False)
        on_disk = build_manifest(exp)
        unlisted = set(on_disk.run_name[on_disk.status == "complete"]) - set(
            man.run_name[man.status == "complete"]
        )
        if unlisted:
            raise ValueError(
                f"{exp.name}: finished runs missing from manifest.csv {sorted(unlisted)}; rebuild it"
            )
        rd = runs_dir(exp)
        for row in man.itertuples():
            if row.status != "complete" or row.split != split or row.kind != kind:
                continue
            if row.sample == "True" and not allow_sample:
                raise ValueError(f"{exp.name}: {row.run_name} is a sample run inside a full experiment")
            run = rd / row.run_name
            output = run / row.output_file
            if sha256_file(output) != row.output_sha256:
                raise ValueError(
                    f"{exp.name}: {row.run_name}/{row.output_file} differs from its manifest sha256"
                )
            cfg = json.loads((run / "config.json").read_text())
            runs.append(cfg | {"run_dir": run, "experiment": exp.name, "records": load_predictions(output)})
    return runs


def sanity_table(exp_dir: Path, run_names: list[str]) -> pd.DataFrame:
    """Pass/fail per run and stage after a sample batch: folder, config, outputs, complete.json, manifest row."""
    man = write_manifest(exp_dir).set_index("run_name")
    rows = []
    for name in run_names:
        run = runs_dir(exp_dir) / name
        cfg_ok = (run / "config.json").exists()
        cfg = json.loads((run / "config.json").read_text()) if cfg_ok else {}
        output = run / cfg.get("output_file", "predictions.jsonl")
        n_pred = sum(1 for _ in output.open()) if output.exists() else 0
        row = {
            "run_name": name,
            "folder": run.is_dir(),
            "config": cfg_ok,
            "outputs": n_pred > 0 and n_pred == cfg.get("n_examples", n_pred),
            "complete": (run / "complete.json").exists(),
            "manifest": name in man.index and man.loc[name, "status"] == "complete",
        }
        row["ok"] = all(v for k, v in row.items() if k != "run_name")
        rows.append(row)
    return pd.DataFrame(rows)
