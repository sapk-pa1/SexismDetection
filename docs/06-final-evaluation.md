# 6 · Final evaluation (`99_final_eval.ipynb`)

The **only** notebook that reads test labels. It scores the frozen zero-shot log-prob batch `20261020_all_all_llms_batch_s42` (15 models × 7 tasks) under a protocol that was written into `docs/experiment-log.md` **before** any label was read. Run name: `20261020_all_all_llms_finaleval_s42`. Runs on CPU.

- **Inputs:** `RUNS_ROOT/*/` (`config.json`, `predictions.jsonl`, `complete.json`) with `RUNS_ROOT = outputs/final_runs_result/outputs`; `data/processed/test/{cmsb,edos}_test.parquet` (labels); `configs/llms.yaml`; the official EXIST validator.
- **Outputs:** `results/final_eval/{tables,figures,paper}/` and `results/final_eval/summary.json` (committed); `OUTPUT_ROOT/20261020_all_all_llms_finaleval_s42/` with `config.json`, `environment.json`, `summary.json`, `README.md` and `evall_submissions/` (~13 MB, so kept out of `results/`).
- No raw post text is printed anywhere.

## Frozen protocol

Chosen by the user on 2026-10-20, recorded before scoring, not to be changed after test scores are seen.

| Item | Rule |
|---|---|
| Soft refusal | A label is refused when its answer mass (P(Yes)+P(No), or summed option-letter mass) is **< 0.5**. A priori, since no dev runs exist |
| Sensitivity | τ ∈ {none, 0.1, 0.5, 0.9}, reported next to the primary number, **never used to pick it** |
| Decisions | Binary/multilabel: P(Yes)/mass ≥ 0.5. Multiclass: argmax over options. No dev tuning |
| Scoring | Strict: a refused label is wrong |
| Fair subset | Posts on which all 15 models answered every label of the task at τ = 0.5 |
| CMSB / EDOS A | Macro-F1 (primary), positive-class F1, accuracy |
| EDOS B / C | Macro-F1 (primary) and accuracy on **gold-sexist** test posts (the SemEval-2023 Task 10 B/C setting); predictions on non-sexist posts ignored |
| EXIST | Test unlabelled → scores from EvALL only. Locally: refusals and inter-model agreement only. Hard and soft files per model; a refused item is left out of the file. 1.3 soft `NO` = Π(1 − P(Yes)) over the 5 labels (independence assumption) |
| Uncertainty | 95% percentile bootstrap CIs, B = 10,000, seed 42, resampling posts |
| Comparisons | Each model vs the task's best model: **paired bootstrap of the macro-F1 difference** (primary); exact McNemar on binary tasks (secondary); Holm correction over the 14 comparisons per task, α = 0.05 |
| CMSB leak rows | Main number includes them; a second number excludes them |

`CONFIG` in the notebook mirrors this (`mass_threshold 0.5`, `sensitivity [None, 0.1, 0.5, 0.9]`, `n_boot 10000`, `seed 42`, `ci_level 0.95`, `alpha 0.05`, `ece_bins 10`, `evall_team "HEFT"` — the team name is a TODO).

Known deviations, stated in the report: no dev tuning of thresholds or the refusal cutoff; one seed and one prompt per task; Qwen3.5-4B thinking-on not run; EXIST English only and pending EvALL.

## Notebook sections

### 1 · Integrity audit (label-free)
`load_finished_runs(RUNS_ROOT, "test")` keeps runs with `complete.json`; models are ordered by model-card size (`n_params` reads the leading number; E4B → 4). For every (task, model) the audit asserts:
- 105 runs = 15 × 7;
- `n_pred == n_examples`, no duplicate ids;
- one `prompt_sha256` per run;
- `revision == revision_requested ==` the registry sha;
- thinking off;
- all 15 models cover the same id set per task.

It also counts label scores exactly equal to 0.5 (ties at the threshold). Output: `tables/audit.csv`.

### 2 · Answer mass and soft refusals (label-free, all 7 tasks)
- `apply_mass_threshold(records, τ)` copies records with `refused = mass < τ` per label (τ = None returns them unchanged). This is done for every τ in the sensitivity grid.
- Per (task, model): median, p05 and min mass; share of posts with any refused label at each τ (a post counts as refused if any of its labels is).
- Per (task, model, label): refusal rate at τ = 0.5 (multilabel per category, since a model may answer one category and refuse another).
- Outputs: `tables/mass_refusal.{csv,tex}`, `tables/refusal_per_label.csv`, `figures/refusal_heatmap.pdf`.

### 3 · Test labels (read here only)
- Reads only `id, sexist, leak_group` (+ `category, vector` for EDOS) from the CMSB and EDOS test files.
- Asserts gold exists for every predicted id and multiclass gold is inside the schema classes. Prints only class counts.
- CMSB uses the annotated-only posts the runs were made on (1,126). EDOS B/C are scored on the **970** gold-sexist test posts.
- `N_LEAK`: 33 CMSB test rows have a `leak_group`, but only **3** are among the scored posts.

### 4 · Main results with CIs and paired tests (CMSB, EDOS A/B/C)
For each labelled task:
1. `strict_arrays(gold, records@τ=0.5, schema)` → `(ids, y, yp, classes)` int-coded. Binary: a refusal becomes the wrong class. Multiclass: a refusal gets code K (= number of classes), which is wrong and belongs to no class.
2. **Cross-check:** `macro_f1_counts(y, yp, K)` must equal `compute_metrics(...)["macro_f1"]` within 1e-12 for every model.
3. One bootstrap index matrix (10,000 × n, seed 42) per task, **shared by all models** so differences are paired.
4. Per model: point macro-F1, 95% percentile CI, accuracy, positive-class F1 (binary), refused-post count, fair-subset size and macro-F1, CMSB macro-F1 without leak rows, Δ vs best with its CI, `p_boot` (two-sided paired bootstrap), `p_mcnemar` (binary).
5. Best = highest point macro-F1. Holm over the 14 others for `p_boot` (and `p_mcnemar` on binary tasks). `tied_with_best` = best, or Holm-adjusted `p_boot ≥ 0.05`. McNemar does **not** decide ties (it tests accuracy); this was corrected after the first draft.
6. A majority-class reference row per task.

Outputs: `tables/main_results_full.csv`, `tables/main_{dataset}_{task}.{csv,tex}`, `figures/main_macro_f1_ci.pdf` (dot + CI per model; hollow = best or not separable; dashed line = majority class).

### 5 · Sensitivity to the refusal cutoff
Strict macro-F1 per model at each τ in the grid ("none" = argmax over answer tokens regardless of mass). The primary number stays at τ = 0.5 whatever this shows. Output: `tables/sensitivity_threshold.{csv,tex}`.

### 6 · Error analysis (aggregate only)
- EDOS B/C: per-class F1 per model; prediction-share bias (predicted share − gold share per class; positive = over-predicted); row-normalised confusion matrices for the 3 best models with a "refused" column. Outputs: `tables/per_class_f1_task_{b,c}.csv`, `tables/pred_share_bias_task_{b,c}.csv`, `figures/per_class_task_{b,c}.pdf`, `figures/confusion_top3_task_{b,c}.pdf`.
- CMSB and EDOS A: predicted vs gold positive rate. Output: `tables/positive_rate.{csv,tex}`.

### 7 · Calibration (binary tasks)
ECE of P(Yes)/mass with 10 equal-width bins over all scored posts (the score is used whether or not the label was refused; binary mass is near 1 for every model). Also: number of distinct score values (shows the bf16 coarse-grid issue), share of scores exactly 0.5, posts without a score, and the signed gap mean score − gold rate. Outputs: `tables/calibration.{csv,tex}`, `figures/calibration_top3.pdf` (reliability curves of the top-3 models).

### 8 · Cross-model agreement (label-free, all 7 tasks)
- Fleiss' κ across the 15 models on every test post, with a refusal as its own category. For EXIST 1.3, κ per label (3 codes: no, yes, refused), then averaged.
- Pairwise percent agreement between models per task.
- Outputs: `tables/fleiss_kappa.csv`, `tables/pairwise_agreement_{dataset}_{task}.csv`, `figures/pairwise_agreement.pdf` (EDOS A and EXIST 1.1).
- κ values are not directly comparable across tasks with different numbers of categories.

### 9 · Model size vs macro-F1 (descriptive)
Spearman ρ between model-card size and strict macro-F1 per labelled task. With 15 models and one seed this is descriptive only. Outputs: `tables/size_vs_f1.csv`, `figures/size_vs_f1.pdf`.

### 10 · EXIST: label-free description and EvALL export
- Per model and EXIST task: answered posts; predicted sexist rate (1.1); class shares (1.2); per-category and `NO` rates (1.3). Output: `tables/exist_prediction_rates.csv`.
- **EvALL files** (`src/utils/evall.py`): for each task × model × variant × {hard, soft}:
  - `strict` variant: records at τ = 0.5; `nocut` variant: τ = None (added because EvALL's handling of missing items is unconfirmed).
  - Items: `{"test_case": "EXIST2025", "id": "<id>", "value": …}`. A post with any refused or zero-mass label is **left out**.
  - Values: 1.1 hard `"YES"/"NO"`, soft `{"YES": p, "NO": 1−p}`; 1.2 hard the argmax class, soft the class probabilities normalised by mass; 1.3 hard the list of predicted categories or `["NO"]`, soft the 5 P(Yes) values plus `NO = Π(1 − P(Yes))`.
  - File name: `{task1_1|task1_2|task1_3}_{hard|soft}_{team}_{run}.json` (5 `_`-separated parts, as the validator splits it; team and run must not contain `_`). Run id = the model short name, or `{model}-nocut`.
  - Each file must pass the official validator's JSON-schema check (`parser_json`), and all ids must be in 600001–600978 (English test).
  - Then `process_format_runs_by_task` is run per task; its log goes to `validator_log_{task}.txt`. The only allowed errors are the **count checks** (English only: 978 of the 2,076 es+en items the validator expects).
  - 180 files in total (15 × 3 × 2 × 2). Manifest with item counts, omitted posts, τ and SHA-256: `evall_submissions/manifest.csv` and `results/final_eval/tables/evall_manifest.csv`.

### 11 · Paper tables
Built only from the frames above with `write_paper_table` and `tex_escape`, into `results/final_eval/paper/`:

| File | Content |
|---|---|
| `models.tex` | model, repo, revision (10 chars), params, licence |
| `main.tex` | macro-F1 and 95% CI per task; bold = best, † = not separable (paired bootstrap, Holm); majority-class row |
| `significance.tex` | Δ vs best, Holm `p_boot`, Holm `p_McN` (binary) |
| `secondary.tex` | CMSB accuracy, F1-sexist, macro-F1 without leak rows; EDOS A accuracy, F1-sexist; EDOS B accuracy and fair-subset macro-F1; EDOS C accuracy |
| `refusal.tex` | non-answer rates: EDOS B/C over the 970 scored posts, EXIST 1.2 over 978, EDOS C over all 4,000 |
| `sensitivity.tex` | EDOS B/C macro-F1 per τ, only models whose score changes |
| `calibration.tex` | predicted positive rate, ECE, signed gap; gold row |
| `agreement.tex` | classes, posts, Fleiss' κ per task |
| `prompts.tex` | appendix: exact system and user messages per task with a placeholder post |

### 12 · Save
`summary.json` (config, models, protocol pointer, every main-results row, κ, size correlations, run names) to both `results/final_eval/` and the run folder; plus `config.json`, `environment.json` and `README.md` in the run folder.

## Metric definitions

- **Macro-F1:** mean over classes of F1 = 2TP / (n_pred + n_true), 0 when both are 0 (sklearn `zero_division=0`). Binary macro-F1 averages the sexist and not-sexist F1. Multiclass macro-F1 is over the K real classes only; the refusal code counts as a wrong prediction and adds to no class's TP.
- **Positive-class F1:** F1 of the sexist class.
- **Percentile CI:** 2.5th and 97.5th percentiles of the 10,000 bootstrap values.
- **Paired bootstrap p:** with d_b = metric_b(model) − metric_b(best) on the shared resamples, p = min(1, 2·min(P(d ≤ 0), P(d ≥ 0))).
- **Exact McNemar:** on per-post correctness of two systems; binomial test on the discordant pairs (scipy `binomtest`).
- **Holm:** step-down adjusted p-values, monotone, clipped at 1, returned in input order.
- **Fleiss' κ:** standard formula over an (items × raters) code matrix.
- **ECE:** Σ_bins (n_bin / n)·|mean score − observed positive rate|, equal-width bins on [0, 1].

The statistics code (`src/utils/stats.py`) was checked: `macro_f1_counts` equals sklearn on random 4- and 11-class data with refusal codes; the same seed gives identical bootstrap results; Fleiss' κ gives 0.210 on the Wikipedia worked example.

## Headline results

Source: `results/final_eval/tables/main_results_full.csv` and the notebook's final summary cell. Test split, zero-shot log-prob, seed 42, one prompt per task, strict scoring at τ = 0.5, macro-F1 with 95% bootstrap CI. These describe this setting only.

| Setting (n scored) | Best model, macro-F1 [95% CI] | Not separable from best (Holm) | Median of 15 | Majority class |
|---|---|---|---|---|
| CMSB binary, annotated (1,126) | Granite-4.1-8B, 0.821 [0.797, 0.843] | Llama-3.1-8B, Qwen3.5-4B, Gemma-4-E4B, Jamba2-3B, SmolLM3-3B | 0.767 | 0.405 |
| EDOS A (4,000) | Gemma-4-E4B, 0.712 [0.696, 0.726] | Jamba2-3B, SmolLM3-3B, Qwen3.5-4B | 0.577 | 0.431 |
| EDOS B, gold-sexist (970) | Falcon-H1-7B, 0.508 [0.466, 0.549] | none | 0.353 | 0.159 |
| EDOS C, gold-sexist (970) | Gemma-4-E4B, 0.256 [0.227, 0.284] | Falcon-H1-7B (0.253) | 0.107 | 0.032 |

Other observations (same sources; see the summary cell of `99` for the full list):
- **Refusals** occur only on lettered-option tasks. EDOS C over the 970 scored posts: Aya-Expanse-8B 98.8%, SmolLM3-3B 97.4%, Granite-4.1-8B 80.9%, Ministral-3-3B 7.3%, Gemma-4-E4B 0.6% (13.5% over all 4,000 posts). EXIST 1.2: Granite 17%. Binary and Yes/No tasks: 0% for every model.
- **Fair subset:** empty for EDOS C; 865 of 970 posts for EDOS B.
- **Leak rows:** excluding the 3 scored CMSB leak rows changes macro-F1 by at most 0.002.
- **Cutoff sensitivity:** none → 0.5 changes no CMSB or EDOS A score. The large moves are on EDOS C (Aya 0.116 → 0.003, SmolLM3 0.109 → 0.009, Granite 0.171 → 0.107), so these models are reported with both numbers.
- **Positive rate:** gold 0.32 (CMSB) and 0.24 (EDOS A). 14 of 15 models over-predict on CMSB and all 15 on EDOS A (0.26–0.76). ECE: CMSB 0.03–0.33, EDOS A 0.09–0.51.
- **Fleiss' κ** across models: CMSB 0.57, EDOS A 0.43, EDOS B 0.20, EDOS C 0.05, EXIST 1.1 0.49, EXIST 1.2 0.12, EXIST 1.3 0.44 (mean over labels).
- **Size:** Spearman ρ 0.37 (CMSB), −0.10 (EDOS A), 0.52 (EDOS B, p = .045 uncorrected), 0.27 (EDOS C); descriptive only.

### Corrections made during review (2026-10-20)
- Leak rows: "33" referred to the whole CMSB test file; only 3 are scored.
- Tie marks had used McNemar on binary tasks; the protocol names the paired bootstrap. After the fix, Qwen3.5-4B is tied with Gemma-4-E4B on EDOS A (Holm p = 0.056); CMSB unchanged.
- EDOS B/C refusal rates in the paper use the 970 scored posts; the all-post rate is a separate column.
- "No exact 0.5 scores" was wrong: one EXIST 1.3 label score of Gemma-4-E4B is exactly 0.5 and is predicted positive (labelled tasks have none).

## EXIST and EvALL

- EXIST 2025 test gold is not public (a web search on 2026-10-20 found no release). EXIST 2025 is listed in EvALL 2.0, so post-campaign scoring appears possible; not yet tested.
- Submission steps: confirm the registered team name (replace `"HEFT"` in `CONFIG["evall_team"]`, no `_`), confirm whether English-only runs are accepted, upload the files, and save the returned scores under the run folder.
- Official EXIST metrics are ICM / ICM-soft (hierarchical for 1.2 and 1.3), with F1 secondary; settings hard-hard and soft-soft. No local ICM is implemented.

## The report (`docs/report/`)
- `llm_baselines_report.tex` (12 pages) and the revised `zero_shot_sexism_llms.tex` (18 pages; after a four-reviewer audit; no repo-internal references and no HEFT). Both `\input` the generated `results/final_eval/paper/*.tex` and the figures.
- Compile: `cd docs/report && tectonic -X compile zero_shot_sexism_llms.tex` (Tectonic 0.17.0).
- Citations were checked against original sources; see the experiment log for which ones and the remaining TODO in `refs.bib`.

## How to rerun
Rerunning `99` on the same inputs must reproduce `results/final_eval/` exactly (deterministic: fixed seed, fixed bootstrap matrix). Never change the protocol values in `CONFIG` after scoring; a new protocol needs a new run name and a new log entry.

## Post-hoc section (addendum A1, added 2026-10-23)

`99` has a new section after section 12, "Post-hoc analyses (addendum A1)" (cells `ph-00` to `ph-18`). It implements [12-protocol-addendum.md](12-protocol-addendum.md). Sections 1–12 are not edited.

**Verification (2026-10-23).** Running the whole notebook with the new section regenerated all 44 non-PDF files of `results/final_eval/` (tables, paper tables, `summary.json`) byte for byte. The 9 PDF figures embed a creation date, so they are regenerated but not byte-identical.

**Before the real run:**
1. Commit `docs/12-protocol-addendum.md`, `configs/lexicons.yaml` and `configs/prompt_variants.lock`; the addendum requires them to be fixed first.
2. Run the batches of [05](05-batch-runs.md#batch-presets-for-addendum-a1-docs12-3), notebook 30 (encoders) and notebook 40 (explanations).
3. Put their experiment folder names into `PH["experiments"]`.

**What it does.**
- The frozen batch is registered read-only as experiment `20261020-000000_experiment_zeroshot-frozen`, and reloaded through its manifest. An assertion checks that this gives exactly the runs and records used by sections 1–12.
- Each section of docs/12 has one cell. A cell whose batch has not run yet prints "skipped".
- Tables go to `results/posthoc_eval/tables/`. The README there lists each table with its status label (PH or PR).
- Per-model failure tables (ids and codes only) and the explanation id lists go to `OUTPUT_ROOT/20261023_all_all_llms_posthoc_s42/`.

**Sanity mode** (`PH["sanity"] = True`):
- B = 200, and the section may read sample experiments.
- Outputs go only to `OUTPUT_ROOT/…-sanity/`, never into `results/`.
- It ends with SANITY OK.
- On 2026-10-23 the whole chain passed on CPU with stub models: every experiment type as a sample experiment, notebooks 30 and 40 as scripts, and `99` twice. The numbers from that check were not read and the outputs were deleted.

