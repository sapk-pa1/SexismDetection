# HEFT — Multi-label Sexism Detection

Research repo for a paper on multi-label sexism detection. Proposed architecture: **HEFT**, built on **NeoBERT**. Compared against vanilla NeoBERT, fine-tuned encoders and five prompted LLMs. Target submission: mid-November 2026.

HEFT in one paragraph: TODO

**Current phase: notebook experimentation.** Experiments run in `notebooks/`. `src/` holds shared LLM code (added on request, 2026-10-19): `src/schemas/` (per-task OutputSchemas, `Example`/`Prediction` records, registry checked against `configs/data_prep.yaml`), `src/prompts/` (one YAML template per task + builder), `src/model/` (loading, Yes/No log-prob scoring, JSON-constrained generation, `infer_single`), `src/utils/` (data loading, seeds/revisions/env info, run-folder I/O, strict metrics; added 2026-10-20). `run_inference.py` (repo root) runs one model over one dev/train split per task. Don't add more to `src/` or create scripts unless asked. `configs/` holds YAML settings for notebooks that have moved theirs out (so far `configs/data_prep.yaml` for 01).

## Layout
```
notebooks/   experiment notebooks (run on Colab from Google Drive)
src/         schemas/, prompts/, model/, utils/: shared LLM inference code
run_inference.py  dataset-level LLM inference script (resumable)
configs/     YAML settings loaded by notebooks (committed)
data/        datasets (gitignored, never commit)
outputs/     per-run predictions, metrics, logs (gitignored)
results/     small final tables and figures worth committing
```

## Environment
- The repo is cloned into Google Drive and run on Google Colab.
- Set paths once, at the top of each notebook: `PROJECT_ROOT`, `DATA_ROOT`, `OUTPUT_ROOT`, `HF_HOME`. Never hard-code paths anywhere else.
- Secrets (`HF_TOKEN`, etc.) come from Colab secrets (`google.colab.userdata`). Never put them in notebook cells or git.
- Pin package versions in the install cell. Olmo 3 needs transformers ≥ 4.57.

## Datasets
- **CMSB**: primary dataset. Binary only (`sexist`).
- **EDOS** (SemEval-2023 Task 10): Task A binary; Task B (4 classes) and Task C (11 classes) are single-label, one class per sexist post. All three are used.
- **EXIST 2025**: tweets only. Task 1.1 binary; Task 1.3 multi-label over 5 categories. Keep both hard (majority) and soft (annotator) labels. English only for now (`languages` in `configs/data_prep.yaml`); Spanish can be added back.
- Each dataset is evaluated on its own tasks; tasks, target columns and class order live under `tasks` in `configs/data_prep.yaml`.
- Use the official splits. Tune everything on dev.

## Models
- HEFT on NeoBERT, plus vanilla NeoBERT with the same head and training budget (the key baseline).
- Encoders: RoBERTa, DeBERTa-v3, HateBERT.
- LLMs, instruct checkpoints only, list frozen as of October 2026:
  - Qwen3.5-9B (the only model run with thinking on and off)
  - Gemma 4 E4B (-it)
  - Ministral 3 8B Instruct 2512 (ships in FP8)
  - Olmo 3 7B Instruct
  - Llama 3.1 8B Instruct
- Record the Hugging Face repo ID and revision for every model in every run.

## Research rules
- **Test set:** `01_data_prep` builds it into `data/processed/test/` (no label stats printed); read it only in `notebooks/99_final_eval.ipynb`. Freeze per-label thresholds on dev first.
- **LLM labels:** score each label as a Yes/No log-probability, not free generation. If generation is unavoidable, use JSON-constrained decoding plus a prefilled reply.
- **Refusals:** log every raw output. Never count a refusal as "no label". Report the refusal rate per model.
- **Every run** writes to `OUTPUT_ROOT/{run_name}/`: the config dict, model revision, seed, predictions (JSONL keyed by example ID) and metrics.
  Run name format: `{date}_{dataset}_{task}_{model}_{method}_s{seed}`
- **Long loops** must be resumable: skip IDs already in the predictions file.
- Keep raw offensive text out of printed cells, commit messages and docs beyond what an analysis needs.

## Do / Don't
- Don't download models or start GPU-heavy runs unless asked.
- Don't commit data, weights, outputs or secrets.
- Ask before adding packages or changing splits, the model list or label definitions.
- Prefer editing an existing notebook over creating a new one.

## Later (not now)
Once experiments settle, move stable notebook code into `src/` and `configs/`.