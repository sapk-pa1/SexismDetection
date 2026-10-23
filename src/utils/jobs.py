"""Batch LLM runs: job list, stable run names, status from run folders, one subprocess per job (optionally several at once)."""

from __future__ import annotations

import itertools
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.registry import LLMSpec


def method_name(
    backend: str,
    thinking: bool,
    split: str,
    limit: int | None,
    shots: int = 0,
    sample: int | None = None,
    variant: str = "",
) -> str:
    """The `{method}` part of a run name, e.g. logprob-zs-dev, logprob-fs1-test or logprob-zs-test-sample-n5.

    `variant` tags prompt/option variants (e.g. "v2", "T3", "perm2"); empty for the frozen v1 prompts.
    """
    method = f"{backend}-{f'fs{shots}' if shots else 'zs'}{'-think' if thinking else ''}"
    method += f"-{variant}" if variant else ""
    method += f"-{split}"
    method += f"-n{limit}" if limit is not None else ""
    return method + (f"-sample-n{sample}" if sample is not None else "")


def variant_tag(
    prompt_variant: str | None = None,
    option_perm: int | None = None,
    subset: int | None = None,
    ids_file: str | None = None,
) -> str:
    """Method-name tag of a run variant, e.g. "v2", "perm2", "sub300", "ids-xai-fnfp"; "" for the frozen setting."""
    parts = [
        f"sub{subset}" if subset else "",
        f"ids-{Path(ids_file).stem}" if ids_file else "",
        prompt_variant or "",
        f"perm{option_perm}" if option_perm else "",
    ]
    return "-".join(p for p in parts if p)


def variant_flags(
    prompt_variant: str | None = None,
    option_perm: int | None = None,
    subset: int | None = None,
    ids_file: str | None = None,
) -> tuple[str, ...]:
    """run_inference flags for one variant (same keys as `variant_tag`)."""
    flags = ["--prompt-variant", prompt_variant] if prompt_variant else []
    flags += ["--option-perm", str(option_perm)] if option_perm else []
    flags += ["--ids-file", str(ids_file)] if ids_file else []
    return tuple(flags + (["--subset", str(subset)] if subset else []))


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
    shots: int = 0  # shots per class from the frozen configs/fewshot file; 0 is zero-shot
    experiment_dir: str = ""  # run folder goes to {experiment_dir}/runs/; empty = OUTPUT_ROOT (old batches)
    sample: int | None = None  # sanity mode: seeded N posts through the whole pipeline
    extra: tuple[str, ...] = ()  # variant flags, e.g. ("--prompt-version", "v2"); part of the method name

    def argv(self) -> list[str]:
        args = ["--dataset", self.dataset, "--task", self.task, "--split", self.split, "--model", self.model]
        args += ["--backend", self.backend, "--seed", str(self.seed), "--run-name", self.run_name]
        args += ["--thinking"] if self.thinking else []
        args += ["--shots", str(self.shots)] if self.shots else []
        args += ["--experiment-dir", self.experiment_dir] if self.experiment_dir else []
        args += ["--sample", str(self.sample)] if self.sample is not None else []
        args += list(self.extra)
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
    shots: int = 0,
    experiment_dir: str = "",
    sample: int | None = None,
    variants: list[dict[str, Any]] | None = None,
    extra_args: tuple[str, ...] = (),
) -> list[Job]:
    """All (model, task, split) jobs in model-major order; thinking models get an extra think-on job.

    `variants` lists run settings with the keys of `variant_tag`, e.g. [{}, {"prompt_variant": "v2"}];
    None runs the frozen setting only. Each variant is its own job and run name. `extra_args` go to
    every job (e.g. ("--max-new-tokens", "32")).
    """
    unknown = sorted(set(models) - set(registry))
    if unknown:
        raise KeyError(f"not in configs/llms.yaml: {unknown}")
    jobs = []
    for model, (dataset, task), split in itertools.product(models, tasks, splits):
        # Thinking on needs the generate backend (log-prob reads the first reply token)
        # Few-shot is log-prob only, so the think-on job stays zero-shot
        modes = [(backend, False, shots)] + ([("generate", True, 0)] if model in thinking_models else [])
        for (be, think, k), v in itertools.product(modes, variants or [{}]):
            method = method_name(be, think, split, limit, k, sample, variant_tag(**v))
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
                    make_run_name(date, dataset, task, model, method, seed),
                    registry[model].vram_gb,
                    registry[model].blocked,
                    k,
                    experiment_dir,
                    sample,
                    variant_flags(**v) + tuple(extra_args),
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
    heartbeat: Callable[[dict[str, Any]], None] | None = None,
    heartbeat_every_s: float = 60.0,
) -> dict[str, tuple[int, Path]]:
    """Run jobs concurrently on one GPU: start in order while the vram_gb estimates fit the budget.

    Each job is still its own batch-1 process, so its scores don't depend on what runs beside it.
    A job larger than the budget runs alone. Interrupting the cell terminates the children; their
    runs stay resumable. `heartbeat` receives a progress dict about every `heartbeat_every_s`.
    """
    pending, running, results = list(jobs), {}, {}
    last_beat = 0.0

    def beat(final: bool = False) -> None:
        failed = sorted(n for n, (code, _) in results.items() if code != 0)
        heartbeat({
            "state": "finished" if final else "running", "n_jobs": len(jobs), "pending": len(pending),
            "running": sorted(running), "done_ok": len(results) - len(failed), "failed": failed,
        })  # fmt: skip

    try:
        while pending or running:
            if heartbeat and time.time() - last_beat >= heartbeat_every_s:
                beat()
                last_beat = time.time()
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
        if heartbeat:
            heartbeat({"state": "interrupted", "n_jobs": len(jobs), "pending": len(pending) + len(running)})
        print("interrupted: running jobs stopped; rerun to resume them")
        raise
    if heartbeat:
        beat(final=True)
    return results


def tail(path: Path, n: int = 5) -> str:
    """Last lines of a log, for failure messages (logs carry ids and scores, not post text)."""
    return "\n".join(path.read_text().splitlines()[-n:])


def jobs_from_ids_dir(
    registry: dict[str, LLMSpec],
    ids_dir: Path,
    *,
    date: str,
    seed: int,
    backend: str,
    prompt_variant: str | None,
    experiment_dir: str = "",
    sample: int | None = None,
    extra_args: tuple[str, ...] = (),
    pattern: str = "*-fnfp.txt",
) -> list[Job]:
    """One job per id list `{dataset}-{task}-{model}-{split}[-fnfp].txt` (written by 99), e.g. the self-explanations.

    Task keys contain no '-', model names may; the split is the last part before any suffix.
    """
    jobs = []
    for f in sorted(Path(ids_dir).glob(pattern)):
        if not f.read_text().strip():
            continue
        parts = f.stem.removesuffix("-fnfp").split("-")
        dataset, task, split, model = parts[0], parts[1], parts[-1], "-".join(parts[2:-1])
        v = {"prompt_variant": prompt_variant, "ids_file": str(f)}
        method = method_name(backend, False, split, None, 0, sample, variant_tag(**v))
        jobs.append(Job(model, dataset, task, split, backend, False, seed, None,
                        make_run_name(date, dataset, task, model, method, seed), registry[model].vram_gb,
                        registry[model].blocked, 0, experiment_dir, sample, variant_flags(**v) + tuple(extra_args)))  # fmt: skip
    return jobs
