# HEFT: multi-label sexism detection

Research code for a paper on sexism detection. The proposed model, **HEFT**, is built on **NeoBERT**. It is compared against vanilla NeoBERT, fine-tuned encoders and prompted LLMs on three English datasets.

> **Status:** notebook experimentation. Experiments live in `notebooks/`; shared LLM inference code (per-task output schemas, prompt templates, backends) is in `src/`. Results are not final. Target submission: mid-November 2026.

**HEFT in one paragraph:** TODO.

## Datasets

Each dataset is evaluated on its own tasks, with the official splits. Tasks, target columns and class order are defined under `tasks` in [`configs/data_prep.yaml`](configs/data_prep.yaml).

| Dataset | Tasks used | Split (train / dev / test) |
|---|---|---|
| **CMSB** ("Call me sexist, but…", Samory et al. 2021) | binary `sexist` | 9,543 / 1,363 / 2,725 (frozen 70/10/20 split) |
| **EDOS** (SemEval-2023 Task 10) | Task A binary; Task B 4 classes; Task C 11 classes (B and C single-label) | 14,000 / 2,000 / 4,000 (official) |
| **EXIST 2025**, tweets, English | Task 1.1 binary; Task 1.2 source intention, 4 classes incl. NO; Task 1.3 multi-label, 5 categories; hard and soft labels | 3,260 / 489 / 978 (official; test unlabelled) |

The datasets are **not redistributed** here. Get them from their original releases (TODO: add links and licence notes) and place them under `data/` like this:

```
data/
  CMSB/sexism_data.csv, sexism_annotations.csv
  EDOS/edos_labelled_aggregated.csv, edos_labelled_individual_annotations.csv
  EXIST/EXIST 2025 Tweets Dataset/{training,dev,test}/*.json
  EXIST/evaluation/golds/          official hard/soft gold files
  splits/cmsb_split_s42.csv        frozen CMSB split (see note)
```

**CMSB split:** `data/splits/cmsb_split_s42.csv` holds ids and split names only. It cannot be regenerated, because the code that made it was lost. `01_data_prep` checks its SHA-256 against [`results/data_manifest.json`](results/data_manifest.json) instead. TODO: say how others can obtain it.

## Models

- **HEFT** on NeoBERT, and **vanilla NeoBERT** with the same head and training budget (the key baseline).
- **Encoders:** RoBERTa, DeBERTa-v3, HateBERT.
- **LLMs:** 15 open-weight instruct checkpoints of at most 8B total parameters, frozen 2026-10-20 with pinned revisions in [`configs/llms.yaml`](configs/llms.yaml). Qwen3.5-4B is run with thinking on and off.
  - LLM labels are scored as Yes/No log-probabilities, not free generation.
  - Refusals are logged per (example, label) and reported per model; they are never counted as a "no".

## Repository layout

```
notebooks/   experiment notebooks, run in order (see below)
src/         schemas/ (per-task OutputSchemas, records), prompts/ (YAML templates), model/ (LLM backends)
configs/     YAML settings loaded by notebooks
docs/        experiment-log.md: dated record of every run and decision
results/     small committed artefacts (data manifest; later, final tables and figures)
data/        datasets (gitignored, never committed)
outputs/     one folder per run (gitignored)
```

## Notebooks

| Notebook | Purpose |
|---|---|
| `00_colab_setup` | Mount Drive, set paths, check the data layout, HF token and GPU |
| `01_data_prep` | Build `data/processed/{train,dev,test}/*.parquet` from the raw files; check every input and output against the manifest |
| `02_data_analysis` | Descriptive statistics of the processed data; writes tables, figures, `analysis.md` and `findings.md` |
| `1x` (planned) | Encoders: baselines, vanilla NeoBERT, HEFT |
| `20_llm_runs` | Batch LLM inference: every model × task × split as a resumable `run_inference.py` job (dry run by default) |
| `2x` (planned) | LLMs: refusal analysis, error analysis |
| `3x` (planned) | Explainability |
| `4x` (planned) | Statistics: bootstrap CIs, significance tests |
| `99_final_eval` (planned) | The only notebook that reads test labels |

## Running

The notebooks are written for **Google Colab** with the repo cloned into Google Drive. They also run locally from `notebooks/` (Python 3.12).

1. Clone the repo into Drive. In each notebook's first cell, set `PROJECT_ROOT` to the clone. All other paths (`DATA_ROOT`, `OUTPUT_ROOT`, `HF_HOME`) derive from it.
2. Put the datasets under `data/` as shown above.
3. Add `HF_TOKEN` as a Colab secret; gated models (Llama) need it. Never paste tokens into cells.
4. Run `00_colab_setup`, then `01_data_prep`, then the experiment notebooks.

Package versions are pinned in each notebook's install cell (`PINS`), and each notebook checks that the installed versions match. Settings for `01` come from `configs/data_prep.yaml`; edit the YAML, not the notebook.

## Reproducibility conventions

- **Run folders.** Every run writes to `outputs/{run_name}/`: config, seed, model repo ID and revision, input hashes, `pip freeze`, predictions keyed by example ID, metrics and a short README.
  - Run names follow `{date}_{dataset}_{task}_{model}_{method}_s{seed}`.
  - A finished run is never overwritten.
- **Thresholds and tuning** are set on dev and frozen before the test set is read.
- **Test labels** are read only in `99_final_eval`. `02_data_analysis` and `run_inference.py --split test` read test files label-blind (text only; no test metrics).
- **Long loops** are resumable: they skip IDs already in the predictions file.
- **Notebook outputs** are stripped before committing, because they can contain offensive text.
- **Logging.** Each run and decision is logged in [`docs/experiment-log.md`](docs/experiment-log.md).

## Results

TODO: no final results yet. See `docs/experiment-log.md` for the current state.

## Citation

TODO.

## Licence

TODO. Dataset licences are set by their original releases.

## Content warning

The datasets contain sexist, offensive and violent language.
