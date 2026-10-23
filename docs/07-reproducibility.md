# 7 · Reproducibility

A result that can't be reproduced from the repo plus `OUTPUT_ROOT/{run_name}/` doesn't count (`.claude/rules/research-integrity.md`). This page lists what is recorded at each stage and how to reproduce everything from scratch.

## What every stage records

| Stage | Run name | Recorded |
|---|---|---|
| `01_data_prep` | `{%Y%m%d-%H%M}_all_prep_none_dataprep_s42` | `config.json` (config, its SHA-256, env, notebook SHA-256), `pip_freeze.txt`, `data_manifest.json` copy, `metrics.json` (processed hashes, written files, audit, stats), `README.md`. Each Parquet footer (`heft` key): pipeline, run name, raw source hashes, env, content hash |
| `02_data_analysis` | `{%Y%m%d-%H%M}_all_analysis_none_eda_s42` | `config.json`, `env.json` (pins, notebook SHA-256, input content hashes), `pip_freeze.txt`, `metrics.json`, CSVs, figures, `analysis.md`, `findings.md`, `README.md` |
| `run_inference.py` | `{date}_{dataset}_{task}_{model}_{method}_s{seed}` | `config.json` (all args, repo, requested and resolved revision, chat options, data file and SHA-256, filters, command), `environment.json` (Python, torch, transformers, CUDA, GPU, git commit and dirty flag, pip freeze), `predictions.jsonl` (prompt SHA-256 per line), `metrics.json` (dev/train), `complete.json` (incl. peak GPU memory), `README.md` |
| `20_llm_runs` | `{date}_all_all_llms_batch_s{seed}` | `config.json`, `status_*.csv` and `failed_*.json` per session, `logs/{run_name}.log` per job, `README.md` |
| `99_final_eval` | `20261020_all_all_llms_finaleval_s42` | `config.json`, `environment.json`, `summary.json`, `README.md`, `evall_submissions/` (+ manifest with SHA-256, validator logs); committed copies in `results/final_eval/` |

## Determinism
- **Seeds:** `set_seeds(seed)` seeds Python `random`, NumPy, PyTorch CPU and CUDA, and `transformers.set_seed`. Every run uses seed 42.
- **Inference** is deterministic by construction: one forward pass per prompt, batch size 1, no sampling. Remaining nondeterminism: GPU kernels (bf16 matmuls) and the PyTorch Mamba fallback for Jamba2/Falcon-H1. Parallel jobs on the same GPU don't share batches, so they don't affect each other's scores.
- **Data prep** has no randomness; row order is fixed by `(split, id)`.
- **Bootstrap** uses `np.random.default_rng(42)`; one index matrix per task shared by all models.
- **LIME** depends on its seed (stored with the result); SHAP on `max_evals`.

## Pinned identities
- **Data:** raw SHA-256s and processed content hashes in `results/data_manifest.json`; each inference run also stores the SHA-256 of the Parquet file it read (`data_sha256`, file bytes).
- **Models:** commit sha per model in `configs/llms.yaml`, resolved again at run time and stored as `revision` next to `revision_requested`; the `99` audit asserts they match.
- **Prompts:** `prompt_sha256` of the template YAML in every prediction; `99` asserts one sha per run.
- **Code:** git commit and dirty flag in `environment.json` (test batch: `d054aee…`, clean).
- **Packages:** pins in `requirements.txt` and `00`; full `pip freeze` per run.

## Reproducing from scratch

1. **Get the data.** Download CMSB, EDOS and EXIST 2025 from their original releases and place them as in [02 §2.1](02-data-pipeline.md#21-raw-inputs). Obtain the frozen `data/splits/cmsb_split_s42.csv` (it can't be regenerated; how others obtain it is still TODO).
2. **Environment.** Python 3.12, `pip install -r requirements.txt` plus Colab's torch; or run `00_colab_setup`.
3. **Processed data.** Run `01_data_prep` with `update_manifest: false`. It must report "raw files match the manifest" and "processed content matches the manifest" and write nothing (or write identical content).
4. **Analysis (optional).** Run `02_data_analysis`; its numbers must match the reference run's `metrics.json`.
5. **Inference.** On a GPU runtime with `HF_TOKEN` (Llama and Aya licences accepted), set `CONFIG["date"]` in `20_llm_runs` to a **new** date (never reuse `20261020`, which would resume/skip the finished runs) and `dry_run: False`. Or run single jobs:
   ```
   python run_inference.py --dataset edos --task task_b --split test --model gemma4-e4b --seed 42
   ```
6. **Evaluation.** Point `RUNS_ROOT` in `99_final_eval` at the new runs, give `CONFIG["run_name"]` a new name, and run it. With the same weights, prompts and data the predictions should match up to GPU numerics, and the tables should match `results/final_eval/`.

To verify the published tables without a GPU, run `99_final_eval` on the existing `outputs/final_runs_result/outputs/` and compare the regenerated `results/final_eval/` with git.

## Guards that enforce the rules
- Finished runs are never overwritten: `prepare_run_dir` refuses a folder with `complete.json`/`metrics.json`; `01` and `02` refuse an existing run folder; `99` writes to fixed paths, so rerunning it regenerates the same files.
- A resumed run must have identical settings (`config.json` diff, except `command`).
- Test labels never load outside `99`: `load_split` reads only id/text columns on test; `02` asserts no label column is in `TEST_COLS`; `01` prints no test statistic.
- Config/schema/template drift fails at load time.
- `01` stops on any hash mismatch unless a rebuild is intended.
- `99` cross-checks its fast metric path against `compute_metrics` for every model and task.
- **(2026-10-23)** Experiment folders:
  - `run_inference.py` refuses `--sample` in a full experiment and a full run in a sample experiment.
  - `open_experiment` refuses to reopen an experiment with changed settings.
  - `load_experiment_runs` refuses sample experiments, finished runs missing from `manifest.csv`, and outputs whose sha256 differs from the manifest.
  - Prompt variants are refused unless their sha256 matches `configs/prompt_variants.lock`.
  - `pytest` (58 tests) checks these guards and that v1 prompts, template sha256s and frozen run names are unchanged.

## Checklist for a new experiment
1. Write the rules first: an addendum section in `docs/12` (or a new addendum) before any run whose results will be reported. Pick an experiment name; run names inside it follow `{date}_{dataset}_{task}_{model}_{method}_s{seed}`. Never reuse a finished one.
2. Run the batch once in sample mode (`sample = N`, SANITY OK), then in full.
3. Put settings in `CONFIG` or a `configs/*.yaml`, not in code cells.
4. Tune only on dev; freeze thresholds before test.
5. Write predictions incrementally, keyed by id, resumable.
6. Save config, seed, model repo and revision, data hash, pip freeze, GPU, git commit.
7. Write a run `README.md`, and append an entry to `docs/experiment-log.md` (date, what and why, run names, key results with source files, deviations, next steps).
8. Report refusals and failed runs; never drop a run silently.
