"""Batch LLM runs: job list, stable run names, status from run folders, one subprocess per job (optionally several at once)."""

from __future__ import annotations

import itertools
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    vram_gb: float = 0.0  # from configs/llms.yaml, for packing parallel jobs
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
                Job(
                    model,
                    dataset,
                    task,
                    split,
                    be,
                    think,
                    seed,
                    limit,
                    name,
                    registry[model].vram_gb,
                    registry[model].blocked,
                )
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


def _launch(job: Job, script: Path, env: dict[str, str], log_dir: Path) -> tuple[subprocess.Popen, Any, Path]:
    """Start one job process; its full output is appended to log_dir/{run_name}.log."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{job.run_name}.log"
    f = log.open("a")
    f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(job.argv())}\n")
    f.flush()
    proc = subprocess.Popen(
        [sys.executable, str(script), *job.argv()],
        stdout=f,
        stderr=subprocess.STDOUT,
        env={**os.environ, **env},
        cwd=script.parent,
    )
    return proc, f, log


def run_job(job: Job, script: Path, env: dict[str, str], log_dir: Path) -> tuple[int, Path]:
    """Run one job in its own process (frees GPU memory, isolates crashes) and wait for it."""
    start = time.time()
    proc, f, log = _launch(job, script, env, log_dir)
    code = proc.wait()
    f.close()
    print(f"  exit {code} after {(time.time() - start) / 60:.1f} min  log: {log.name}")
    return code, log


def run_jobs_parallel(
    jobs: list[Job],
    script: Path,
    env: dict[str, str],
    log_dir: Path,
    *,
    gpu_budget_gb: float,
    max_parallel: int,
    poll_s: float = 5.0,
) -> dict[str, tuple[int, Path]]:
    """Run jobs concurrently on one GPU: start in order while the vram_gb estimates fit the budget.

    Each job is still its own batch-1 process, so its scores don't depend on what runs beside it.
    A job larger than the budget runs alone. Interrupting the cell terminates the children; their
    runs stay resumable.
    """
    pending, running, results = list(jobs), {}, {}
    try:
        while pending or running:
            for name, (proc, f, log, _job, t0) in list(running.items()):
                if proc.poll() is not None:
                    f.close()
                    results[name] = (proc.returncode, log)
                    del running[name]
                    print(f"  done  {name}: exit {proc.returncode} after {(time.time() - t0) / 60:.1f} min")
            used = sum(r[3].vram_gb for r in running.values())
            # First fit in job order; a big job waits until enough memory frees up
            for job in list(pending):
                if len(running) >= max_parallel:
                    break
                if used + job.vram_gb <= gpu_budget_gb or not running:
                    proc, f, log = _launch(job, script, env, log_dir)
                    running[job.run_name] = (proc, f, log, job, time.time())
                    used += job.vram_gb
                    pending.remove(job)
                    print(
                        f"  start {job.run_name} (~{job.vram_gb:g} GB, {used:g}/{gpu_budget_gb:g} GB in use)"
                    )
            time.sleep(poll_s)
    except KeyboardInterrupt:
        for proc, f, *_ in running.values():
            proc.terminate()
            proc.wait()
            f.close()
        print("interrupted: running jobs stopped; rerun to resume them")
        raise
    return results


def tail(path: Path, n: int = 5) -> str:
    """Last lines of a log, for failure messages (logs carry ids and scores, not post text)."""
    return "\n".join(path.read_text().splitlines()[-n:])
