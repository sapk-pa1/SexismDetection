"""Run folders: guarded creation, resumable predictions.jsonl and the per-run README."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def prepare_run_dir(run_dir: Path, config: dict[str, Any], ignore: tuple[str, ...] = ("command",)) -> None:
    """Create a run folder or reopen an unfinished one with the same config; never touch a finished run."""
    if run_dir.exists():
        # complete.json marks every finished run (test runs have no metrics.json); metrics.json covers older runs
        if (run_dir / "complete.json").exists() or (run_dir / "metrics.json").exists():
            raise FileExistsError(f"{run_dir.name} is finished; use a new run name")
        old = json.loads((run_dir / "config.json").read_text())
        diff = sorted(k for k in config.keys() - set(ignore) if old.get(k) != config[k])
        if diff:
            raise ValueError(f"{run_dir.name} exists with different settings: {diff}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))


def read_done(pred_path: Path) -> set[str]:
    """IDs already predicted; drops a truncated last line left by a disconnect."""
    if not pred_path.exists():
        return set()
    lines = pred_path.read_text().splitlines()
    done, good = set(), []
    for i, line in enumerate(lines):
        try:
            done.add(json.loads(line)["id"])
            good.append(line)
        except (json.JSONDecodeError, KeyError):
            if i != len(lines) - 1:
                raise ValueError(f"{pred_path}: corrupt line {i + 1} is not the last line") from None
            print(f"dropping truncated last line of {pred_path.name}")
            pred_path.write_text("".join(f"{g}\n" for g in good))
    return done


def load_predictions(pred_path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in pred_path.read_text().splitlines()]


def write_readme(run_dir: Path, config: dict[str, Any], status: str, metrics: dict[str, Any] | None) -> None:
    """Short per-run README: purpose, command, config summary and outcome."""
    lines = [
        f"# {config['run_name']}",
        "",
        f"**Purpose.** Prompted-LLM inference ({config['backend']}) for {config['dataset']}/{config['task']} "
        f"on {config['split']}.",
        f"**Command.** `{config['command']}`",
        f"**Model.** `{config['model_id']}` @ `{config['revision']}`; seed {config['seed']}; "
        f"thinking {config['thinking']}; mass threshold {config['mass_threshold']}.",
        f"**Data.** `{config['data_file']}` (sha256 `{config['data_sha256'][:12]}…`), {config['n_examples']} examples "
        f"({config['filters']}).",
        f"**Status.** {status}",
    ]
    if config.get("shots"):
        lines.insert(-1, f"**Few-shot.** {config['shots']} per class from `{config['shots_file']}`.")
    if metrics is not None:
        keys = ["n_scored", "macro_f1", "micro_f1", "f1_positive", "accuracy", "exact_match"]
        summary = ", ".join(
            f"{k} {metrics[k]:.4g}" if isinstance(metrics[k], float) else f"{k} {metrics[k]}"
            for k in keys
            if k in metrics
        )
        lines.append(f"**Outcome.** {summary} ({metrics['scoring']}; full numbers in `metrics.json`).")
    elif status == "finished":
        lines.append("**Outcome.** Predictions only (label-blind test run); scored in `99_final_eval`.")
    (run_dir / "README.md").write_text("\n".join(lines) + "\n")


def load_finished_runs(root: Path, split: str = "test") -> list[dict[str, Any]]:
    """Every finished run under root for one split: config fields, run_dir and its prediction records."""
    runs = []
    for cfg_path in sorted(root.glob("*/config.json")):
        cfg = json.loads(cfg_path.read_text())
        if cfg.get("split") != split or not (cfg_path.parent / "complete.json").exists():
            continue
        records = load_predictions(cfg_path.parent / "predictions.jsonl")
        runs.append(cfg | {"run_dir": cfg_path.parent, "records": records})
    return runs
