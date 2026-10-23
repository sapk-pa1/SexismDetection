# 8 · Code reference

Public helpers are re-exported from the package `__init__.py`; import them from there (`from src.utils import load_split`). Modules that load before `HF_HOME` is set (`src/utils/*`) import torch/transformers inside functions.

## `run_inference.py`
Entry point for one (model, task, split) run. Flags and steps: [04 §4.5](04-llm-inference.md#45-run_inferencepy). Paths from env vars `DATA_ROOT`, `OUTPUT_ROOT`, `HF_HOME`.

## `src/schemas/`

| Name | Module | Purpose |
|---|---|---|
| `OutputSchema` | `base.py` | Base pydantic model; ClassVars `dataset, task, task_type, column`; `prompt_id()`, `labels()`, `json_schema()`, `from_prediction()`, `value()` |
| `TaskType` | `base.py` | `Literal["binary", "multiclass", "multilabel"]` |
| `CmsbBinary, EdosTaskA, EdosTaskB, EdosTaskC, ExistTask11, ExistTask12, ExistTask13`, `ALL_SCHEMAS` | `tasks.py` | One schema per task |
| `Example`, `LabelScore`, `Prediction` | `records.py` | Input post; per-label score; JSONL record |
| `TASKS`, `get_schema(dataset, task)` | `registry.py` | Lookup by (dataset, task) |
| `check_against_config(path)` | `registry.py` | Fail on type/column/class-order drift from `configs/data_prep.yaml` |

## `src/prompts/`

| Name | Purpose |
|---|---|
| `PromptTemplate.load(schema)` | Load `templates/{dataset}/{task}.yaml`, check labels match the schema and the needed template (`yes_no` or `options`) exists; store its SHA-256 |
| `PromptTemplate.yes_no_messages(text, label, shots=())` | System + user messages for one Yes/No question, after any shot turns |
| `PromptTemplate.option_messages(text, shots=())`, `option_letters()` | Lettered-option question; letters A… in schema order; shot answers become letters |
| `select_shots`, `freeze_shots`, `load_shots`, `shots_path`, `Shot` (`fewshot.py`) | Seeded, stratified train shots; frozen (IDs only) under `configs/fewshot/` |
| `PromptTemplate.json_messages(text)` | Generate-backend request |
| `ChatOptions(prefill, chat_kwargs)` | Per-model options from `configs/llms.yaml` |
| `render(tokenizer, messages, prefill, enable_thinking, chat)` | Apply the chat template; prompt ends where the reply starts |
| `OPTION_LETTERS`, `TEMPLATE_DIR` | Constants |

## `src/model/`

| Name | Module | Purpose |
|---|---|---|
| `load_hf_model(repo_id, revision, dtype, device_map, token, fp8_dequantize, trust_remote_code)` | `loading.py` | Pinned-revision bf16 load with causal-LM → image-text-to-text fallback |
| `YES_VARIANTS`, `NO_VARIANTS`, `answer_token_ids` | `logprob.py` | Single-token answer variants |
| `next_token_probs(model, tokenizer, prompt)` | `logprob.py` | Softmax over the vocabulary at the reply position |
| `score_yes_no`, `score_options`, `score_example` | `logprob.py` | Log-prob scoring → `LabelScore`s |
| `JsonGenerator(model, tokenizer, schema)` | `generate.py` | JSON-constrained greedy generation (outlines + xgrammar) |
| `infer_single(model, tokenizer, example, schema, *, model_id, revision, backend, template, generator, mass_threshold, top_k, max_new_tokens, enable_thinking, chat)` | `llm.py` | One post → one `Prediction` |

## `src/utils/`

### `data.py`
- `load_split(data_root, dataset, split, schema, all_cmsb=False, limit=None) -> (df, path, filters)`: processed rows for one run; label-blind on test; CMSB annotated-only; EDOS B/C gold-sexist on train/dev; ids as str.

### `repro.py`
- `get_hf_token()`: env, then Colab secrets.
- `resolve_revision(repo_id, revision, token)`: branch/tag → commit sha.
- `set_seeds(seed)`: Python, NumPy, torch (CPU/CUDA), transformers.
- `sha256_file(path)`.
- `environment_info(project_root)`: Python, platform, torch, transformers, CUDA, GPU, git commit/dirty, pip freeze.

### `run_io.py`
- `prepare_run_dir(run_dir, config, ignore=("command",))`: create, or reopen an unfinished run with the same config; refuse finished runs.
- `read_done(pred_path)`: done ids; drops a truncated last line.
- `load_predictions(pred_path)`.
- `write_readme(run_dir, config, status, metrics)`.
- `load_finished_runs(root, split="test")`: every finished run's config + `run_dir` + records.

### `metrics.py`
- `answers(rec, schema)`: per-label answer, None = refused/zero-mass/unparsed.
- `mass_summary(records, threshold)`: per-label mass stats and refusal rate.
- `compute_metrics(df, records, schema, *, split, backend, mass_threshold)`: strict metrics ([04 §4.7](04-llm-inference.md#47-metrics-on-devtrain-compute_metrics)).
- `apply_mass_threshold(records, threshold)`: copies with `refused = mass < threshold`.
- `refusal_rates(records, threshold)`: per-label refused share.
- `answered_ids(records, schema)`: ids with every label answered.
- `strict_arrays(df, records, schema) -> (ids, y, yp, classes)`: int-coded gold and strict predictions (binary/multiclass).
- `UNANSWERED`: multiclass refusal sentinel.

### `stats.py`
- `macro_f1_counts(y, yp, n_classes)`: sklearn-equal macro-F1; codes ≥ n_classes are wrong.
- `bootstrap_indices(n, n_boot, seed)`, `bootstrap_metric(y, yp, n_classes, idx, metric="macro_f1"|"accuracy")`.
- `percentile_ci(samples, level=0.95)`, `paired_bootstrap_p(diff)`.
- `mcnemar_exact(y, a, b) -> (a_only, b_only, p)`, `holm(pvals)`.
- `fleiss_kappa(codes, n_categories)`, `expected_calibration_error(y, score, n_bins=10) -> (ece, rows)`.

### `evall.py`
- `evall_items(records, schema, soft)`: `[{test_case, id, value}]` for answered posts.
- `write_evall(items, out_dir, task, soft, team, run)`: `{task}_{hard|soft}_{team}_{run}.json`.
- `TEST_CASE = "EXIST2025"`, `EVALL_TASK` (task key → EvALL task id).

### `report.py`
- `set_style()`, `cmap(kind)`: one matplotlib style and palette for all figures.
- `save_table(df, out_dir, stem, caption)`: `stem.csv` (full precision) + booktabs `stem.tex`.
- `ci_dotplot(ax, df, title, ref, ref_label)`, `heatmap(ax, mat, title, kind, vmin, vmax, fmt)`.
- `tex_escape(s)`, `write_paper_table(path, header, rows, caption, label, colspec, …)`.

### `jobs.py`
- `method_name(backend, thinking, split, limit)`, `make_run_name(date, dataset, task, model, method, seed)`.
- `Job` (one `run_inference.py` call; `argv()`), `build_jobs(...)`.
- `job_status(output_root, job)`, `status_table(output_root, jobs)`.
- `run_job(job, script, env, log_dir)`, `run_jobs_parallel(jobs, script, env, log_dir, *, gpu_budget_gb, max_parallel, poll_s=5)`, `tail(path, n=5)`.

### `registry.py`
- `LLMSpec` (frozen dataclass), `load_llm_registry(path)`.

### `explain.py`
- `LabelScorer`, `shap_explain`, `lime_explain` ([04 §4.8](04-llm-inference.md#48-attribution-helpers-no-runs-yet)).
- Added 2026-10-23 (docs/12 §3.8):
  - `faithfulness`: ERASER comprehensiveness and sufficiency of the top-k SHAP words vs random words, with "..." masking and with deletion
  - `attribution_agreement`: SHAP–LIME ρ and top-3 overlap
  - `lime_stability`: ρ across LIME seeds
  - `lexicon_class`
  - `counterfactual_texts`: gender swap; slurs and profanity removed
  - `code_rationales`: lettered-option log-prob coding of self-explanations.

### `experiments.py` (added 2026-10-23)
Experiment folders ([05](05-batch-runs.md#experiment-folders-sample-mode-and-background-runs-since-2026-10-23)):
- `open_experiment`, `find_experiment`, `register_experiment`, `read_experiment`, `runs_dir`, `experiment_date`
- `build_manifest` / `write_manifest`
- `write_heartbeat`
- `load_experiment_runs`: the only way `99`'s post-hoc section reads runs; it checks manifest rows and output sha256 and refuses sample experiments
- `sanity_table`.

### `scoring_extra.py` (added 2026-10-23, docs/12 §2.2–2.5, §3.1)
- `label_scores`: per-post log-odds, P(Yes)/mass, mass and top token, or answer-forced class shares
- `binary_metrics`: AUROC, AUPRC, Brier and MCC with paired bootstrap draws
- `per_class_f1`, `multilabel_macro_f1`, `macro_f1_2d`, `soft_losses` (L1, Brier, CE)
- `prior_matched`, `batch_calibrated`: label-free decision rules
- `risk_coverage`
- dev-fitted rules `fit_threshold`, `fit_temperature`, `fit_platt`
- `pride_prior`, `pride_debias` (Zheng et al. 2024).

### `failure.py` (added 2026-10-23, docs/12 §2.6–2.8)
- `failure_table`: outcome and reason code per (post, label): non_answer, confident_no/yes, borderline_no/yes, exact_tie; for options also confident_wrong, near_tie, tie_break
- `token_class`: first-token class of non-answers
- `load_lexicons` (`configs/lexicons.yaml`), `word_hits`, `gender_swap`, `remove_words`
- `rate_table`, `mean_ci`
- `contrast_pairs`, `contrast_metrics`: CMSB adversarial pairs.

### `stats.py` additions (2026-10-23, docs/12 §2.1, §3.6)
`simultaneous_ci`, `format_p`, `mde`, `rank_stats`, `spearman_ci`, `approx_randomization_p`, `kendalls_w`, `variance_shares`, `pigeonhole_bootstrap`.

### `evall.py` additions (2026-10-23, docs/12 §2.9, §3.2)
- `evall_items(..., parent_records)`: hierarchical 1.3 decoding, where soft NO is the model's 1.1 P(No) and categories count only when 1.1 says Yes. Without parents the frozen rule is unchanged.
- `HIERARCHY`, `gold_subset`
- `pyevall_scores`: official ICM, ICM-Norm, F1, ICM-Soft and CE via PyEvALL 0.2.11.

### `src/model/text.py`, `src/model/encoder.py` (added 2026-10-23)
- `generate_text`, `parse_option`, `parse_yes_no`: the text backend and its fixed parser.
- `EncoderHP`, `train_encoder`: plain PyTorch fine-tuning with dev selection; label-blind test predictions.
- `encoder_arrays`.

## Configs

| File | Read by | Content |
|---|---|---|
| `configs/data_prep.yaml` | `01`, `02`, `run_inference.py` (`check_against_config`) | data-prep settings and the `tasks` spec |
| `configs/llms.yaml` | `run_inference.py`, `20`, `99` | the 15-model registry |

## Where helpers may live
Scripts and notebooks are entry points (config, paths, main loop, saving). Helpers move into `src/` once a file has more than 3 helpers or ~100 lines of helper code, a helper is needed in 2+ places, or it is research logic (filters, metrics, refusals, seeds, run I/O). A move must not change behaviour; run the same smoke test before and after and log it (`.claude/rules/code-structure.md`). `01` (normalisation, schema building, hashing, writing; ~8 helpers) and `02` (`md_table`, `text_stats`, …) still hold their own helpers, which is over the 3-helper limit; see [09](09-known-issues.md).
