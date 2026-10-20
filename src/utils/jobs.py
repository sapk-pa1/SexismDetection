"""Batch LLM runs: job list, stable run names, status from run folders, one subprocess per job."""

from __future__ import annotations

import itertools
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.registry import LLMSpec


def method_name(backend: str, thinking: bool, split: str, limit: int | None) -> str:
    """The `{method}` part of a run name, e.g. logprob-zs-dev or logprob-zs-test-n5."""
    method = f"{backend}-zs{'-think' if thinking else ''}-{split}"
    return method + (f"-n{limit}" if limit is not None else "")


def make_run_name(date: str, dataset: str, task: str, model: str, method: str, seed: int) -> str:
    """Run name format from CLAUDE.md: {date}_{dataset}_{task}_{model}_{method}_s{seed}."""
    return f"{date}_{dataset}_{task}_{model}_{method}_s{seed}"


@dataclass(frozen=True)
class Job:
    """One run_inference call; `run_name` is fixed up front so a rerun resumes the same folder."""

    model: str
    dataset: str
    task: str
    split: str
    backend: str
    thinking: bool
    seed: int
    limit: int | None
    run_name: str
    blocked: str = ""

    def argv(self) -> list[str]:
        args = ["--dataset", self.dataset, "--task", self.task, "--split", self.split, "--model", self.model]
        args += ["--backend", self.backend, "--seed", str(self.seed), "--run-name", self.run_name]
        args += ["--thinking"] if self.thinking else []
        return args + (["--limit", str(self.limit)] if self.limit is not None else [])


def build_jobs(
    registry: dict[str, LLMSpec],
    models: list[str],
    tasks: list[tuple[str, str]],
    splits: list[str],
    *,
    date: str,
    seed: int,
    backend: str = "logprob",
    limit: int | None = None,
    thinking_models: tuple[str, ...] = (),
) -> list[Job]:
    """All (model, task, split) jobs in model-major order; thinking models get an extra think-on job."""
    unknown = sorted(set(models) - set(registry))
    if unknown:
        raise KeyError(f"not in configs/llms.yaml: {unknown}")
    jobs = []
    for model, (dataset, task), split in itertools.product(models, tasks, splits):
        # Thinking on needs the generate backend (log-prob reads the first reply token)
        variants = [(backend, False)] + ([("generate", True)] if model in thinking_models else [])
        for be, think in variants:
            name = make_run_name(date, dataset, task, model, method_name(be, think, split, limit), seed)
            jobs.append(
                Job(model, dataset, task, split, be, think, seed, limit, name, registry[model].blocked)
            )
    return jobs


def job_status(output_root: Path, job: Job) -> tuple[str, int]:
    """(status, n predictions so far): blocked, done, partial (resumable) or todo."""
    run_dir = output_root / job.run_name
    pred = run_dir / "predictions.jsonl"
    n = sum(1 for _ in pred.open()) if pred.exists() else 0
    if job.blocked:
        return "blocked", n
    if (run_dir / "complete.json").exists() or (run_dir / "metrics.json").exists():
        return "done", n
    return ("partial" if run_dir.exists() else "todo"), n


def status_table(output_root: Path, jobs: list[Job]) -> pd.DataFrame:
    rows = []
    for j in jobs:
        status, n = job_status(output_root, j)
        rows.append({"model": j.model, "dataset": j.dataset, "task": j.task, "split": j.split,
                     "backend": j.backend, "thinking": j.thinking, "status": status, "n_pred": n,
                     "run_name": j.run_name})  # fmt: skip
    return pd.DataFrame(rows)


def run_job(job: Job, script: Path, env: dict[str, str], log_dir: Path) -> tuple[int, Path]:
    """Run one job in its own process (frees GPU memory, isolates crashes); full output to a log file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{job.run_name}.log"
    start = time.time()
    with log.open("a") as f:
        f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(job.argv())}\n")
        f.flush()
        proc = subprocess.run(
            [sys.executable, str(script), *job.argv()],
            stdout=f,
            stderr=subprocess.STDOUT,
            env={**os.environ, **env},
            cwd=script.parent,
        )
    print(f"  exit {proc.returncode} after {(time.time() - start) / 60:.1f} min  log: {log.name}")
    return proc.returncode, log


def tail(path: Path, n: int = 5) -> str:
    """Last lines of a log, for failure messages (logs carry ids and scores, not post text)."""
    return "\n".join(path.read_text().splitlines()[-n:])
