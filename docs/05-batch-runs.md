# 5 · Batch LLM runs (`20_llm_runs.ipynb`)

The notebook builds every (model, task, split) job from its `CONFIG` and runs each as a separate `run_inference.py` **subprocess**. One process per job means each job loads its model fresh, GPU memory is freed afterwards, and one crash doesn't stop the batch. Helpers are in `src/utils/jobs.py`. The notebook only orchestrates; metrics live in each run's folder.

## Paths on Colab
- `OUTPUT_ROOT` on Drive, so runs survive disconnects.
- `HF_HOME = /content/hf_cache` (local disk): Drive is slow for multi-GB reads and 15 models need about 150 GB.
- `data/processed` is copied to `/content/data/processed` and `DATA_ROOT` points there.
- Paths and `HF_TOKEN` are passed to each job via environment variables.

## `CONFIG` (the only cell to edit between batches)

Since 2026-10-23 every batch runs inside one **experiment folder** (below). The 2026-10-20 batch predates this and used `date`/`run_name` keys; it is registered read-only as an experiment by `99`.

| Key | Default | Meaning |
|---|---|---|
| `experiment` | `"zeroshot-main"` | Experiment name (lowercase, digits, `-`). The folder is `OUTPUT_ROOT/experiments/{YYYYMMDD-HHMMSS}_experiment_{name}`. **Rerunning with the same name resumes it**; a changed setting is refused (use a new name). |
| `purpose`, `protocol_ref` | TODO / `docs/12…` | One line on what the batch is for, and the addendum section it runs; stored in `experiment.json` |
| `sample` | `5` | **Sanity mode**: N random posts per job (seeded, label-blind) through the whole pipeline into a separate `{name}-sample` experiment. Run it, check **SANITY OK**, then set `None` for the full batch |
| `models` | `"all"` | Every model in `configs/llms.yaml`, or a list of short names |
| `tasks` | the 7 tasks | CMSB binary; EDOS A, B, C; EXIST 1.1, 1.2, 1.3 |
| `splits` | `["test"]` | `["dev"]` for the dev sweep (allowed for pre-registered secondary analyses since 2026-10-23, docs/12) |
| `backend` | `logprob` | `text` = greedy free text read by the fixed parser (format check, self-explanations) |
| `variants` | `None` | List of run settings with keys `prompt_variant`, `option_perm`, `subset`, `ids_file`; one job and run name per variant (presets below) |
| `extra_args` | `[]` | Flags for every job, e.g. `["--max-new-tokens", "32"]` |
| `ids_dir`, `ids_prompt_variant` | `None`, `"X"` | Self-explanations: one job per `*-fnfp.txt` id list written by `99` (model, task and split come from the file name) |
| `limit` | `None` | First N rows, debugging only (adds `-n{N}`) |
| `seed` | 42 | |
| `shots` | `0` | Shots per class from `configs/fewshot` (not run under addendum A1) |
| `thinking_models` | `[]` | Think-on runs disabled |
| `max_parallel`, `gpu_budget_gb` | 7, 70 | Parallel packing on one A100 by `vram_gb` |
| `heartbeat_s` | 60 | How often `heartbeat.json` is refreshed |
| `dry_run`, `stop_on_error` | `False` | Print the job table only / stop at the first failure (sequential mode) |

Scheduling keys (`max_parallel`, `gpu_budget_gb`, `heartbeat_s`, `dry_run`, `stop_on_error`) may change between sessions without making a new experiment.

## Experiment folders, sample mode and background runs (since 2026-10-23)

```
OUTPUT_ROOT/experiments/{YYYYMMDD-HHMMSS}_experiment_{name}/
  experiment.json   purpose, protocol section, git commit, mode (full | sample), full CONFIG; written once
  runs/{run_name}/  one folder per job (config, predictions.jsonl, complete.json, environment.json, README)
  manifest.csv      one row per run (model, task, split, method, variant, seed, status, output sha256), rebuilt
                    from the run folders after every session; 99 reads runs only through it
  logs/             one log per job
  status_*.csv, failed_*.json   per session, never overwritten
  heartbeat.json    progress (jobs done/running/failed, last update), refreshed during the batch
  README.md         generated summary
```

- `src/utils/experiments.py`: `open_experiment` (create or resume), `write_manifest`, `write_heartbeat`, `load_experiment_runs` (used by `99`; refuses sample experiments, sample runs, finished runs missing from the manifest, and outputs whose sha256 differs), `register_experiment` (describes an old batch without moving any file), `sanity_table`.
- **Sample mode never mixes with results.** Sample runs go to `{name}-sample` experiments, their run names end in `-sample-n{N}`, `run_inference.py` refuses `--sample` in a full experiment (and a full run in a sample experiment), and `99` refuses sample experiments unless its own sanity mode is on.
- **Background runs.** The batch writes logs and the heartbeat to Drive, so progress can be checked from Drive without the notebook. A closed tab or disconnect loses at most the post in flight; rerun with the same `experiment` name to resume.
- **Recommended workflow:** `sample = 5` → Run all → SANITY OK → `sample = None` → Run all (background) → check `heartbeat.json`.

## Batch presets for addendum A1 (docs/12 §3)

Each row is one experiment (one `CONFIG`); models are `"all"` and the seed 42 unless stated. Run the sample version first.

| `experiment` | `tasks` | `splits` | `backend` | `variants` / other keys |
|---|---|---|---|---|
| `devsweep` (§3.1–3.2) | all 7 | `["dev"]` | `logprob` | `None` |
| `format-check` (§3.3) | EDOS B, EDOS C, EXIST 1.2 | `["test"]` | `text` | `[{"subset": 300}]`, `extra_args = ["--max-new-tokens", "32"]` |
| `answer-prefix` (§3.4) | EDOS B, EDOS C, EXIST 1.2 | `["test"]` | `logprob` | `[{"prompt_variant": "v2"}]` |
| `option-order` (§3.5) | EDOS B, EXIST 1.2 | `["test"]` | `logprob` | `[{"option_perm": k} for k in (1, 2, 3)]` |
| `option-order-pride` (§3.5) | EDOS C | `["test"]` | `logprob` | `[{"subset": 200}] + [{"subset": 200, "option_perm": k} for k in range(1, 11)]` |
| `prompt5` (§3.6) | all 7 | `["test"]` | `logprob` | `[{"prompt_variant": v} for v in ("T1", "T2", "T3", "T4")]` (T0 = the frozen runs) |
| `xai-selfexpl` (§3.8) | from the id lists | from the id lists | `text` | `ids_dir = "20261023_all_all_llms_posthoc_s42/xai_ids"`, `extra_args = ["--max-new-tokens", "64"]` |

Then set the folder names in `99`'s `PH["experiments"]` (`option_order` takes both option-order experiments as a list). Encoders run in notebook 30 and explanations in notebook 40, with the same experiment layout and sample mode.

## Jobs, run names and status

- `build_jobs(registry, models, tasks, splits, date, seed, backend, limit, thinking_models, shots, experiment_dir, sample, variants, extra_args)` makes jobs in model-major order (× variants). A model in `thinking_models` gets an extra `generate` + `--thinking` job. `jobs_from_ids_dir` makes one job per id list instead.
- In an experiment, `{date}` in run names is the experiment's creation date, so run names stay fixed for the experiment's life.
- Each job's `run_name` is fixed up front with `make_run_name(date, dataset, task, model, method, seed)`, e.g. `20261020_edos_task_b_gemma4-e4b_logprob-zs-test_s42`. Since the date is fixed (no time stamp, unlike a direct `run_inference.py` call), a rerun points at the same folders.
- `job_status` per job:
  - `blocked`: the registry entry has a `blocked` reason;
  - `done`: `complete.json` (or `metrics.json` for older runs) exists;
  - `partial`: the folder exists without completion → resumed;
  - `todo`: no folder.
- Only `todo` and `partial` jobs are launched. A partial job resumes from its `predictions.jsonl` (same args + `--run-name`).

## Parallel scheduling (`run_jobs_parallel`)

- Jobs start **in order** while the sum of running jobs' `vram_gb` estimates fits `gpu_budget_gb`, up to `max_parallel`. A job larger than the budget runs alone once nothing else is running (first fit; a big job waits for memory).
- Each job is still a **batch-1** process, so its scores don't depend on what runs beside it.
- The parent polls every 5 s and prints start/done lines with memory in use.
- Interrupting the cell (KeyboardInterrupt) terminates all children; their runs stay `partial` and resumable.
- Each job's full output is appended to `logs/{run_name}.log`. On failure the last 5 lines (`tail`) are reported; logs carry ids and scores, not post text.

After each session the notebook writes `status_{stamp}.csv` and `failed_{stamp}.json` (never overwriting earlier ones) and refreshes the batch `README.md`.

## Decision: test only (2026-10-20)

The user decided to run the LLMs **on test only**, with no dev runs, because evaluation is on test and train/dev are reserved for the supervised models. Consequences, recorded before any test scoring:
- Per-label thresholds and the soft-refusal cutoff cannot be tuned on dev; they were fixed a priori ([06](06-final-evaluation.md#frozen-protocol)).
- EXIST test is unlabelled, so EXIST scores come only from EvALL.
- The CMSB dev run of Qwen3.5-4B remains the only dev result.
- **Partly reversed 2026-10-23 (decision D2):** label-blind dev inference is allowed for the pre-registered secondary analyses of docs/12 (dev-tuned rules, EXIST dev scoring). The primary numbers stay test-only and unchanged.

## The 2026-10-20 batch (`20261020_all_all_llms_batch_s42`)

Hardware: A100-SXM4-80GB, `max_parallel` 7, `gpu_budget_gb` 70, logprob, seed 42.

1. **First session** (old registry: Granite Guardian blocked, Jamba not yet added). 91 of 98 launched jobs exited 0; 13 models finished all 7 tasks. All 7 Aya Expanse jobs failed after 0.3 min with `403 ... Cannot access gated repo` (licence not accepted on the token's account; an access problem, not a code failure). Job times ranged from 1.2 min (CMSB, Qwen) to 21.7 min (EDOS C, Nemotron); Falcon-H1 was slower than other ~8B models (13.6 min on EDOS A vs 6.0–8.0).
2. **Second session**, after accepting the Aya licence and syncing the new registry: the 7 Aya and 7 Jamba2 jobs ran.

Final state: **105 finished test runs** (15 models × 7 tasks), unpacked under `outputs/final_runs_result/outputs/` (also as `result_final.zip`). The batch folder holds `status_20261020-062514.csv` from an earlier planning session (210 jobs, dev and test, 196 todo / 14 blocked) and 106 job logs; the later session's status table is not in the folder. A correction note was appended to the batch README by the `99` audit; nothing else in the folder was changed.

Per-task example counts on test: CMSB 1,126 (annotated only), EDOS 4,000 (all posts, every task), EXIST-en 978.

`git diff d054aee d758689 -- src run_inference.py` is empty, so all 105 runs used the same inference code (the test runs record `git_commit d054aee…`, `git_dirty: false`).

## Known Colab pitfalls (from the log)
- The kernel can hold an old numpy in memory after the pins installed a new one (`_blas_supports_fpe` errors): restart the session.
- The notebook ends with two ad-hoc diagnostic cells (versions in the kernel; whether the Drive copy has the new registry, `jobs.py`, the 1.2 template and the `intention` column) and two empty cells. They are not part of the pipeline.
- **Security:** the saved notebook's secrets cell must read `userdata.get("HF_TOKEN")`. See [09](09-known-issues.md).
