# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Zero-shot sexism detection with small open LLMs (OSNEM submission)

Research repo for a paper on sexism detection with 15 prompted open LLMs (≤ 8B) on CMSB, EDOS and EXIST 2025: binary detection, fine-grained single-label categories (EDOS B/C, EXIST 1.2 intention) and the multi-label EXIST 1.3 task. It includes failure analysis ("why a model does not say Yes") and explainability. Target: Elsevier *Online Social Networks and Media*, submission mid-November 2026. Fine-tuned encoders (DeBERTa-v3, RoBERTa) are the supervised reference.

**Scope decision (user, 2026-10-23, D1):** this paper is the standalone zero-shot study. **HEFT** (on NeoBERT) and vanilla NeoBERT move to a follow-up paper; HEFT's description is still TODO.

**Current phase: notebook experimentation.** Experiments run in `notebooks/`; shared LLM inference code is in `src/` and is driven by `run_inference.py`. Don't create new scripts unless asked; when a script or notebook collects too many helpers, move them into `src/` (see `.claude/rules/code-structure.md`).

## Layout
```
notebooks/   experiment notebooks (run on Colab from Google Drive); conventions in notebooks/CLAUDE.md
src/         schemas/, prompts/, model/, utils/: shared LLM inference code
run_inference.py  dataset-level LLM inference script (resumable)
configs/     YAML settings (committed): data_prep.yaml, llms.yaml, fewshot/, lexicons.yaml (failure analysis and counterfactuals), prompt_variants.lock (frozen variant hashes)
docs/        README.md + 01–12: full pipeline documentation (10 evaluation review, 11 OSNEM roadmap, 12 protocol addendum A1); experiment-log.md: dated record of every run and decision; report/: LaTeX reports
tests/       pytest suite (CPU only; stub models, no downloads)
data/        datasets (gitignored, never commit); layout and quirks in data/CLAUDE.md
outputs/     per-run predictions, metrics, logs (gitignored)
results/     small final tables and figures worth committing (data_manifest.json; final_eval/ and posthoc_eval/ from 99_final_eval)
```
Research-integrity, comment and code-structure rules live in `.claude/rules/`.

## Commands
Local env is `.venv/` (Python 3.12); on Colab, packages come from the 00 install cell. Pins are in `requirements.txt` and that cell (the `requirements.lock` comment in `pyproject.toml` is stale).
- Lint / format: `ruff check src run_inference.py tests`, `ruff format src run_inference.py tests` (config in `pyproject.toml`).
- Tests: `.venv/bin/python -m pytest tests -q`. There are 58 tests, all CPU and offline, using a stub model and cached tokenizers.
- Smoke run: `python run_inference.py --dataset cmsb --task binary --model gemma4-e4b --limit 5` (`--model` is a name in `configs/llms.yaml`; `--model-id/--model-name/--revision` override it)
  - Other flags:
    - `--split {dev,train,test}`: test is label-blind (predictions only, no metrics).
    - `--backend {logprob,generate,text}`, `--thinking`, `--mass-threshold`, `--all-cmsb`, `--seed`.
    - `--shots K`: frozen train shots from `configs/fewshot/`, logprob only.
  - Added 2026-10-23 (docs/04 §4.9):
    - `--experiment-dir`, `--sample N` (sanity mode), `--subset N`, `--ids-file`
    - `--prompt-variant {v2,T1..T4,X}`: must match `configs/prompt_variants.lock`
    - `--option-perm K`
  - Paths come from env vars `DATA_ROOT`, `OUTPUT_ROOT`, `HF_HOME` (default: repo dirs). `HF_TOKEN` from env or Colab secrets.
- Batch runs: `notebooks/20_llm_runs.ipynb` builds every (model, task, split, variant) job from `CONFIG` and runs each as a `run_inference.py` subprocess (helpers in `src/utils/jobs.py`).
  - Each batch is one experiment folder, `OUTPUT_ROOT/experiments/{YYYYMMDD-HHMMSS}_experiment_{name}/`, holding `runs/`, `manifest.csv`, `logs/` and `heartbeat.json` (`src/utils/experiments.py`).
  - The same `experiment` name resumes it.
  - Run `CONFIG["sample"] = N` first (SANITY OK), then `None`.
  - Presets per batch are in docs/05.
  - Notebooks 30 (encoders) and 40 (explainability) use the same layout.
- Resume: same args plus `--run-name <existing>`. Refused if the run is finished (`complete.json`, or `metrics.json` for older runs) or its settings changed.

## Architecture (LLM pipeline)
- Flow: `utils.data.load_split` (processed parquet, eval filters) → `schemas.Example.from_row` (models see `text_norm`, not `text`) → `schemas.get_schema(dataset, task)` → `prompts.PromptTemplate.load(schema)` (YAML template, sha256 stored per prediction) → `model.infer_single` → one `Prediction` JSONL line → `utils.metrics.compute_metrics` (strict: unanswered = wrong).
- Backends: `logprob` (primary) scores Yes/No per label for binary/multilabel, and one lettered-option prompt for multiclass (EDOS B/C); `generate` is JSON-constrained (outlines + xgrammar) and cannot show refusals. Thinking mode needs `generate`, since log-prob reads the first reply token.
- Class order is defined in three places that must match: `tasks` in `configs/data_prep.yaml`, `src/schemas/tasks.py`, `src/prompts/templates/{dataset}/{task}.yaml`. `check_against_config` and `PromptTemplate.load` fail on drift; a new task touches all three.
- Refusals: `LabelScore.refused` stays None until a dev-frozen mass threshold is passed; `parsed` is None if any label is refused or has zero mass.
- `HF_HOME` must be set before transformers is imported, so `src/utils` imports torch/transformers inside functions.
- A run folder holds `config.json` (incl. per-model chat options), `environment.json` (pip freeze, GPU, git commit), `predictions.jsonl`, `complete.json`, `README.md`, and `metrics.json` on dev/train only.
- `configs/llms.yaml` (via `utils.registry`) holds repo, pinned sha and per-model options: `prefill`, `chat_kwargs`, `fp8_dequantize`, `trust_remote_code`, and `blocked` (run_inference refuses blocked models; none are blocked now).
- Known issue: bf16 logits quantize P(Yes), giving a coarse score grid and exact ties (first CMSB dev Gemma run; on options the earlier-shown class wins a tie). New runs store `tie` and `rendered_sha256` per score. Float32 last-position scoring is pending.
- **Prompt variants** (`{task}.{variant}.yaml`: v2 answer prefix, T1–T4 rewordings and formats, X self-explanation) keep the v1 definitions byte-identical. They load only if their sha256 is in `configs/prompt_variants.lock`; v1 files and their hashes must never change.

## Environment
- The repo is cloned into Google Drive and run on Google Colab.
- Set paths once, at the top of each notebook: `PROJECT_ROOT`, `DATA_ROOT`, `OUTPUT_ROOT`, `HF_HOME`. Never hard-code paths anywhere else.
- Secrets (`HF_TOKEN`, etc.) come from Colab secrets (`google.colab.userdata`). Never put them in notebook cells or git.
- Pin package versions in the install cell. Olmo 3 needs transformers ≥ 4.57.

## Datasets
- **CMSB**: primary dataset. Binary only (`sexist`).
- **EDOS** (SemEval-2023 Task 10): Task A binary; Task B (4 classes) and Task C (11 classes) are single-label, one class per sexist post. All three are used.
- **EXIST 2025**: tweets only. Task 1.1 binary; Task 1.2 source intention, single-label over NO/REPORTED/JUDGEMENTAL/DIRECT (all tweets, NO is a class); Task 1.3 multi-label over 5 categories. Task keys follow the official numbering (`task_1_1`, `task_1_2`, `task_1_3`); quote `"NO"` in YAML (bare NO parses as false). Keep both hard (majority) and soft (annotator) labels. English only for now (`languages` in `configs/data_prep.yaml`); Spanish can be added back.
- Each dataset is evaluated on its own tasks; tasks, target columns and class order live under `tasks` in `configs/data_prep.yaml`.
- Use the official splits. Tune everything on dev. Since 2026-10-23 (D2) the zero-shot LLMs may run label-blind on dev for pre-registered secondary analyses (docs/12); the primary LLM numbers stay test-only. EXIST is scored on its labelled English dev split (official golds, PyEvALL), because the test gold is not public and English-only EvALL scoring is unconfirmed.
- The CMSB split (`data/splits/cmsb_split_s42.csv`) cannot be regenerated; 01 checks its hash against `results/data_manifest.json`.

## Models
- LLMs: 15 open-weight instruct checkpoints, each <= 8B total params (model-card figure), frozen 2026-10-20 in `configs/llms.yaml` with pinned shas. All runs so far have thinking off; a Qwen3.5-4B think-on run needs a generate-then-score path and is not planned for this paper. gpt-oss is excluded (21B total). Granite Guardian 3.3-8B (a guard model) was replaced by AI21 Jamba2-3B on 2026-10-20.
- Encoder reference (notebook 30, docs/12 §3.7): DeBERTa-v3-base and RoBERTa-base, 3 seeds, dev-selected. HateBERT is not run.
- HEFT on NeoBERT, plus vanilla NeoBERT: follow-up paper (D1).
- Record the Hugging Face repo ID and revision for every model in every run.

## Research rules
- **Test set:** `01_data_prep` builds it into `data/processed/test/` (no label stats printed); test labels are read only in `notebooks/99_final_eval.ipynb`. Freeze per-label thresholds on dev first. Label-blind reads of test text are allowed in `02_data_analysis`, `run_inference.py --split test`, notebook 30 (encoder test predictions) and notebook 40 (explanations; it receives id lists only, never outcome cells).
- **Post-hoc rule:** test labels were read on 2026-10-20. Every later analysis is PH (post-hoc secondary) or PR (pre-registered), and its rules go into `docs/12-protocol-addendum.md` (or a new addendum) and are committed **before** it runs. `99`'s post-hoc section and new experiments run only after that commit.
- **LLM labels:** score each label as a Yes/No log-probability, not free generation. If generation is unavoidable, use JSON-constrained decoding plus a prefilled reply.
- **Refusals:** log every raw output. Never count a refusal as "no label". Report the refusal rate per model.
- **Every run** writes to its run folder (inside an experiment folder since 2026-10-23): the config dict, model revision, seed, predictions (JSONL keyed by example ID) and metrics. Sample (sanity) runs live only in `-sample` experiments and never reach evaluation.
  Run name format: `{date}_{dataset}_{task}_{model}_{method}_s{seed}`
- **Long loops** must be resumable: skip IDs already in the predictions file.
- Keep raw offensive text out of printed cells, commit messages and docs beyond what an analysis needs.

## Do / Don't
- Never run `git commit`, `git push` or anything else that writes git history; the user commits. When work is done, suggest commit messages instead.
- Don't download models or start GPU-heavy runs unless asked.
- Don't commit data, weights, outputs or secrets.
- Ask before adding packages or changing splits, the model list or label definitions.
- Prefer editing an existing notebook over creating a new one.
