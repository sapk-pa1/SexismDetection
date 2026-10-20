# Experiment log

## 2026-10-19 · 01_data_prep: per-dataset tasks, audit flags, run-record fixes
**What and why.** Review of `notebooks/01_data_prep.ipynb`. The processed files forced one multi-label layout on all datasets, but each dataset has its own tasks: CMSB binary; EDOS Task A binary, Task B 4-class, Task C 11-class (B/C single-label); EXIST 1.1 binary, 1.3 multi-label. Changes:
- `tasks` block in `configs/data_prep.yaml` (type, target column, class order); 01 asserts stored values match it.
- Per-dataset schemas. CMSB: dropped the always-null `labels`, the content-code `soft`/`votes` and the source `subset` column (not used). EDOS: `category` (B) and `vector` (C, official codes) as single strings, plus `{category,vector}_{soft,votes}` from the 3 annotators; new assert that Task C sits under Task B in both files. EXIST: `soft`/`votes` renamed `labels_soft`/`labels_votes`.
- Text audit flags on all datasets (`empty_norm`, `dup_group`, `leak_group`). Flagged only; nothing removed.
- CMSB stratification also checked on annotated-only rows (within 1 point); sexist rates printed for train/dev only (the test rate was printed before).
- Run record: run name now has date and time (`%Y%m%d-%H%M`) and an existing run folder stops the run; adds `pip_freeze.txt`, notebook SHA-256 and `README.md`.

**Runs.**
- `20261019-1518_all_prep_none_dataprep_s42`: `update_manifest: true`; rewrote `results/data_manifest.json` (all 9 processed hashes changed because of the schema change; raw hashes unchanged).
- `20261019-1519_all_prep_none_dataprep_s42`: `update_manifest: false`; raw and processed content match the manifest, no files written.
- `20261019_all_prep_none_dataprep_s42` (earlier, old schema) is kept as history; its `data_manifest.json` is the previous manifest.

**Key results** (`outputs/20261019-1518_all_prep_none_dataprep_s42/metrics.json`, train/dev only for labels):
- CMSB sexist rate 0.1328 train / 0.1328 dev; annotated-only 0.3204 / 0.3198.
- CMSB flags: 67 / 12 / 16 placeholder-only inputs (train/dev/test); 108 / 19 / 33 cross-split `leak_group` rows (unchanged from before); 8 duplicate groups in train/dev with conflicting `sexist`.
- EXIST (en): 2 train and 1 test rows share a `text_norm` across splits (new finding; official split, so flagged only). EDOS: no duplicates.
- EDOS hard label disagrees with the 2-of-3 annotator majority on 772 of 16,000 train/dev posts (unchanged).

**Issues / deviations.** Run name format changed from `{date}` to `{date}-{HHMM}` so same-day reruns can't overwrite a run.

**Next steps.** Decide which flagged CMSB rows to exclude per experiment; decide how EDOS B/C are evaluated (gold-sexist subset vs end to end; check the task paper); check EvALL scoring for an English-only EXIST submission.

## 2026-10-19 · 01_data_prep: data analysis (no run)
**What and why.** Full analysis of `notebooks/01_data_prep.ipynb` against the raw files in `data/`. Label statistics use train/dev only (EDOS, EXIST); CMSB checks use no split labels because the split file is missing (below).

**Blocking issue.** `data/splits/cmsb_split_s42.csv` and `data/processed/` were removed between 15:19 and 15:38. No copy was found on local disk or Google Drive. The manifest pins the split (sha256 `7680236e…`, 147,023 bytes). 01 can't run without it, and the split must not be regenerated without a decision.

**Findings** (ad-hoc checks, not a pipeline run):
- Normalisation: no leftover URLs or `@handles` in any dataset; 287 EXIST (en) tweets had HTML entities (unescaped); no NFC changes. Two CMSB rows keep a glued `RTMENTION3377:` because `USER_RE` requires a word boundary before `MENTION`.
- CMSB format cues: `RT` prefix in 8.4% of annotated vs 0.3% of `other` posts; hashtags 21.0% vs 43.7%. Extra source cues for the all-rows setting.
- Length (words in `text_norm`, p99/max): CMSB 28/41, EDOS 48/58, EXIST en 55/63.
- EXIST en: 1.1 hard gold for 2,870 / 444 train/dev tweets, 1.3 hard for 2,864 / 444; no 1.1 vs 1.3 contradictions; 6 train tweets have 1.1 = YES but no 1.3 majority. `n_annot` for 1.3 is 4–6 in train, always 6 in dev.
- EDOS train/dev, 3,884 sexist posts: no annotator majority for Task B on 963 and Task C on 1,452; the majority differs from the aggregated label on 163 (B) and 62 (C); 385 are labelled sexist with only 1 of 3 annotators voting sexist.

**Update (same day).** A byte-identical copy of the split (sha256 matches the manifest) was found in another Claude session's temporary scratchpad and copied back to `data/splits/cmsb_split_s42.csv`. `data/processed/` is still missing; rerun 01 in check mode to rebuild it.

**Next steps.** Rerun 01 (`update_manifest: false`) to rebuild `data/processed/`. Then decide on the `MENTION` regex fix and whether to strip `RT [USER]:` prefixes; both change `text_norm` and need `update_manifest: true`.

## 2026-10-19 · 01_data_prep: code review (no run, no code changes)
**What and why.** Review of `notebooks/01_data_prep.ipynb` for bugs. Notebook unchanged since runs `20261019-1518…`/`-1519…` (sha256 `678545bdc00e` matches both run records).

**Findings.**
- Still blocking: `data/splits/cmsb_split_s42.csv` and `data/processed/` are missing, so the notebook can't run.
- EXIST `n_annot` counts Task 1.3 annotators (UNKNOWN dropped), while `sexist_soft` is the 1.1 soft gold over 6. They differ on 53 English train tweets (52 with 5 and 1 with 4 for 1.3). Don't compute 1.1 vote counts as `sexist_soft * n_annot`.
- EXIST 1.1 hard gold has no count assert; a value other than YES/NO would become null without an error. It's YES/NO only in the current release.
- `RTMENTION3377:` is left unnormalised (open from the analysis entry).
- 00 doesn't pin `pyyaml`, although 01 says it uses "the same pins as 00".
- The Findings cell doesn't mention the missing split file.

**Next steps.** Restore the split file; decide on the fixes above (any `text_norm` change needs `update_manifest: true`).

## 2026-10-19 · 01_data_prep: review fixes (no run)
**What.** Fixed items 2, 4 and 5 from the code review above. None of them changes the processed data, so the manifest needs no update.
- 01 EXIST cell: asserts that Task 1.1 hard golds are only YES/NO, and checks the 1.1 hard count before the language filter (train 6,064, dev 934, test 0; it matches the gold files).
- 00: added the `pyyaml==6.0.3` pin so it matches 01's `PINS`.
- 01 Findings cell: added the missing split file as a blocker.

**Check.** I ran the edited EXIST cell on its own against the raw files and all its asserts passed. I didn't run the full notebook because the CMSB split file is still missing. The notebook's sha256 changes, so the next run's record will show a different `notebook_sha256` from `20261019-1518…`/`-1519…`.

**Still open.** Restore the split file; decide on EXIST `n_annot` vs `sexist_soft` and on the `RTMENTION` regex.

## 2026-10-19 · 01_data_prep: data fixes from the review (no run; manifest rebuild pending)
**What and why.** Fixed review items 1 and 3. Both change processed content, so the manifest needs a rebuild.
- Item 1, EXIST annotator counts: `n_annot` is now the Task 1.1 annotator count behind `sexist_soft`. This is the same meaning as in CMSB (5) and EDOS (3). The new `labels_n_annot` holds the Task 1.3 count behind `labels_votes`. A new assert checks that 1.1 YES votes / `n_annot` equals the official 1.1 soft gold exactly (the existing assert does the same for 1.3).
- Item 3, glued retweet mentions: `GLUED_RT_RE` turns `RTMENTION12:` into `RT [USER]:`, matching the other retweets.

**Check** (cells run on their own against the raw files; the full notebook can't run without the CMSB split file):
- EXIST: all asserts pass. English train/dev `n_annot` is always 6, and `n_annot != labels_n_annot` on 53 tweets. Test counts are null. The EXIST tables build under the new schema.
- CMSB: `text_norm` changes on 2 rows. EDOS and EXIST: 0 rows change. CMSB `dup_group`/`leak_group` counts may shift once the full pipeline is run.

**Expected effect.** All 3 EXIST tables (new column) and any CMSB table holding the 2 rows will get new hashes. The current manifest no longer matches. With `update_manifest: false`, a run stops at the processed-content check, as intended.

**Next steps.** Restore `data/splits/cmsb_split_s42.csv`, run once with `update_manifest: true`, then once with `false`, and log both runs. Downstream code should use `labels_n_annot` (not `n_annot`) for EXIST 1.3 vote counts.

## 2026-10-19 · 20_llm_logprob: LLM Yes/No log-prob pipeline (no GPU run yet)
**What and why.** New `notebooks/20_llm_logprob.ipynb` and `configs/llm_logprob.yaml`. This is the zero-shot LLM baseline, which scores one prompt per (post, label) from the next-token P(Yes)/P(No), with no free generation (research rule).
- First model: Gemma 4 E4B, the smallest on the list. Repo `google/gemma-4-E4B-it`, revision `ee0ef6023621cff504d758262d4e04895a5af4a2` (main on 2026-10-19), bf16, thinking off.
- Smoke setting: EXIST Task 1.3, dev, 5 examples (seeded sample, seed 42), 25 prompts.
- Prompts: the EXIST category definitions are adapted from `EXIST2025_Annotation_guidelines_Tweets.pdf`. CMSB and EDOS definitions are TODO.
- Each (id, label) pair logs P(Yes), P(No), their mass, P(Yes)/(P(Yes)+P(No)) and the top 10 next tokens to `predictions.jsonl`. The loop resumes by skipping pairs already in the file. `refusal_mass_threshold` is TODO (to be picked on dev).
- Pins: transformers 5.20.1 added to 00's pins. torch is Colab's build and is logged, not pinned.

**Deviation.** The request was to run inference on the test set. The notebook only allows `split: train|dev`, because the repo rule says test is read only in `99_final_eval`. The smoke test runs on dev instead.

**Check (local, no model weights).** The config, data, prompt and tokenizer cells ran on CPU. All 10 Yes/No variants are single tokens. The rendered prompt ends with `<|turn>model\n`, so the next token is the answer. Padding is on the left. `Gemma4ForConditionalGeneration.forward` accepts `logits_to_keep` (checked in the transformers 5.20.1 source). The model forward pass has not run yet.

**Next steps.** Run on a Colab GPU. Check the Yes/No mass and the top-k tokens, then log the run here.

## 2026-10-19 · 02_data_analysis: descriptive analysis of processed data
**What and why.** New `notebooks/02_data_analysis.ipynb`, a descriptive analysis of the processed CMSB, EDOS and EXIST files for train, dev and test.
- Test is opened **label-blind**. Only `TEST_COLS` (ids, `text_norm`, `lang`, audit flags, CMSB `of_id`) are read, so test labels never reach memory.
- CMSB `annotated` / `n_annot` are left out of test because they give away the all-non-sexist `other` source.
- **Deviation:** the repo rule says test is read only in `99_final_eval`. This notebook reads label-free test columns, as the user asked.
- Added pin: matplotlib 3.11.2.

**Run.** `20261019-1618_all_analysis_none_eda_s42`. Run locally on CPU. A scratch run beforehand gave an identical `metrics.json`. Input content hashes are in `env.json`.

**Key results** (files in the run folder; the full list is in the notebook's Findings cell):
- Sexist rate, train / dev (`binary.csv`):
  - CMSB 13.3% (32.0% on annotated posts).
  - EDOS 24.3%.
  - EXIST 39.6% / 43.7% of hard-labelled tweets (390 / 45 have no 1.1 majority).
- Unanimous votes: CMSB 66% / 62%, EDOS 78% / 80%, EXIST 33% / 35%.
- EDOS hard label vs the 2-of-3 majority: they differ on 697 / 14,000 train and 75 / 2,000 dev posts (`binary_soft_vs_hard.csv`). This is consistent with the 95.2% figure in `data/README.md`.
- EDOS Task C dev has fewer than 10 examples for 1.1 (8), 3.3 (9) and 3.4 (7) (`edos_task_c.csv`).
- EDOS: the hard Task B class has at least 2 of the 3 votes for only 70.8% / 72.2% of sexist posts (`edos_agreement.csv`).
- EXIST 1.3 train: 32% of hard-labelled sexist tweets have one category, and 2 categories is the most common count (46%). P(STEREOTYPING | IDEOLOGICAL) = 0.63 and P(OBJECTIFICATION | SEXUAL-VIOLENCE) = 0.61 (`exist_*.csv`).
- Median words, not sexist vs sexist, train: CMSB 16 vs 15, EDOS 22 vs 24, EXIST 26 vs 26 (`length_by_label_train.csv`).

**Next steps.** Decide on CMSB duplicates and leak rows, decide on EDOS Task C thresholds for the tiny dev classes, and state which EXIST subset each metric uses.

## 2026-10-19 · 02_data_analysis: report files (analysis.md, findings.md)
**What.** 02 now writes two report files to its run folder:
- `analysis.md`: every table as Markdown, plus links to the figures.
- `findings.md`: fixed wording, with every number computed from the run's tables, so the text can't drift from the data.

The notebook's static Findings cell now points to `findings.md`.

**Run.** `20261019-1623_all_analysis_none_eda_s42` (new run; `20261019-1618…` is kept unchanged). The numbers are the same as in the 1618 run. One new computed check: every vote share maps to a single hard label for CMSB and EXIST, but not for EDOS.

## 2026-10-19 · .gitignore update (no run)
Extended `.gitignore`:
- More secrets: key/credential files, `kaggle.json`.
- More weight formats: `.pth`, `.gguf`, `.onnx`, `.h5`.
- Experiment-tracker folders, `.claude/settings.local.json`, build/coverage files, editor and OS files.

Checked with a scratch git dir. Only code, configs, docs, notebooks and `results/` would be tracked. `.env`, `data/`, `outputs/` and `hf_cache/` are ignored. Open question: whether to commit `data/splits/cmsb_split_s42.csv` (ids and split only, and it can't be regenerated).

## 2026-10-19 · src/model/llm.py: single-example HF inference (no GPU run)
**What and why.** The user asked for a script that runs inference on a single data point, placed in `src/model/llm.py`. This is the first file in `src/`, created at the user's request. The module scores labels as Yes/No next-token log-probs (no free generation, per the research rules):
- `load_hf_model(repo_id, revision)`: pinned revision, bf16. Falls back to `AutoModelForImageTextToText` for checkpoints with no causal-LM mapping (Ministral 3 / `mistral3` in transformers 5.20.1).
- `score_label`: logs P(Yes), P(No), their mass, P(Yes)/mass and the top-k next tokens. Low mass marks a soft refusal; the threshold is still TODO (to be picked on dev).
- `infer_single(model, tokenizer, id, text, labels)`: one record per post, keyed by ID, with one entry per label.

**Check (local, CPU, tokenizer only).** Using the cached Gemma 4 E4B tokenizer (rev `ee0ef602…`), the prompt ends with `<|turn>model\n`, and all 5 Yes and 5 No variants are single tokens. Ruff passes. No model forward pass has been run.

**Issues.** `notebooks/20_llm_logprob.ipynb` and `configs/llm_logprob.yaml` (logged above) are missing from disk. The run folder `20261019_exist_task_1_3_gemma4-e4b_logprob-zs-dev-n5_s42` is empty. The prompt wording in `llm.py` is new, not taken from that notebook. Label definitions are passed in by the caller. Qwen3.5 with thinking on can't be scored from the first token and needs a separate path (TODO).

**Next steps.** Run one dev example on a Colab GPU and check the Yes/No mass. Then add a resumable loop that writes `predictions.jsonl`.

## 2026-10-19 · src/: per-task OutputSchemas, prompt templates, LLM backends (no GPU run)
**What and why.** At the user's request, split the single `src/model/llm.py` into a package. Each task now has its own output schema, which drives both backends. This replaces the generic one-label prompt from the entry above.
- `src/schemas/`: one pydantic `OutputSchema` per task.
  - `CmsbBinary`, `EdosTaskA`, `ExistTask11`: `sexist: bool`.
  - `EdosTaskB`/`EdosTaskC`: a `Literal` over the classes.
  - `ExistTask13`: 5 bools, with the official names as JSON keys.
  - `Example`, `LabelScore` and `Prediction` are the JSONL records. `registry.check_against_config` fails if a schema's type, column or class order drifts from `configs/data_prep.yaml`.
- `src/prompts/templates/{dataset}/{task}.yaml` plus `builder.py`. Each prediction stores the template's sha256.
- `src/model/`:
  - `logprob.py` (primary): Yes/No per label for binary/multilabel. For multiclass (EDOS B/C), one prompt with lettered options, scored over the option-letter tokens and argmaxed (user's choice).
  - `generate.py` (fallback): JSON-constrained decoding from `schema.json_schema()`, using outlines' xgrammar backend and plain HF `generate`. The grammar starts after the prefill.
  - `llm.infer_single(...) -> Prediction` is the entry point.
- **Definitions.** EXIST is adapted from `EXIST2025_Annotation_guidelines_Tweets.pdf`. EDOS is from Kirk et al. 2023 (arXiv 2303.04222); several definitions were excerpted in our copy and are marked TODO. CMSB is adapted from Samory et al. 2021, Table 2 and Sec. 2.2 (arXiv 2004.12764).
- **Pins** (00 install cell and `requirements.in`): pydantic 2.13.5, outlines 1.3.3, xgrammar 0.2.8 (packages approved by the user), plus transformers 5.20.1, the version already in use. pydantic 2.14.x was skipped because it is less than 2 weeks old.

**Check (CPU; scratch scripts, not committed).**
- `ruff check src` passes.
- The registry matches the config, and a reversed EDOS B class order is caught.
- With the Gemma 4 E4B tokenizer (rev `ee0ef602…`), every task's prompt ends at `<|turn>model\n`. Yes/No and the letters A–K are single tokens.
- Schemas accept their own dump and reject an extra key or an empty object.
- `Example`s were built from 3 dev rows per dataset (text not printed), and `Prediction` JSON round-trips.
- Both backends ran end to end on a tiny random-weight Llama with the Gemma tokenizer; this checks the mechanics only. All 6 tasks gave a Prediction. In generation, EXIST 1.3 hit `max_new_tokens` and was stored as a parse error, as designed.

**Issues / limits.**
- Constrained decoding can't show a refusal, so the log-prob mass stays the refusal signal.
- `pred` uses a provisional 0.5 cutoff on P(Yes)/mass until per-label thresholds are tuned on dev.
- The xgrammar backend can't restrict JSON whitespace (`whitespace_pattern` is unsupported).
- Option-order position bias isn't handled.
- The EDOS B/C evaluation scope is still open.
- The full prompt text is no longer stored per label; it is rebuilt from the template sha256 plus the example ID.

**Next steps.** Fill in the excerpted EDOS definitions from the paper. Run 1 dev example per task with Gemma 4 E4B on Colab and check the Yes/No and option mass. Then add the resumable dev loop in a `2x` notebook.

## 2026-10-20 · run_inference.py: dataset-level LLM inference script (no GPU run)
**What and why.** The user asked for a script in the repo root that runs inference over a dataset. `run_inference.py` runs `src.model.infer_single` over one processed split for one (dataset, task):
- **Data.** Reads `processed/{train,dev}` only and refuses test, which stays in `99_final_eval`. Filters: CMSB keeps annotated rows only (`--all-cmsb` keeps all). EDOS B/C use the gold-sexist subset; their scope is still TODO. `--limit N` takes the first N rows.
- **Model.** The revision is resolved to a commit sha via `huggingface_hub`. `--thinking` needs `--backend generate`.
- **Outputs.** `OUTPUT_ROOT/{run_name}/` gets `config.json` (args, resolved sha, data sha256, filters), `environment.json` (pip freeze, GPU, torch/transformers, git commit and dirty flag), `predictions.jsonl` (flushed per line), `metrics.json` and `README.md`.
- **Resume.** Rerunning with `--run-name` skips done IDs and drops a truncated last line. It refuses a finished run (one with `metrics.json`) or a run with different settings.
- **Metrics.** Strict scoring: refused, zero-mass or unparsed answers count as wrong.
  - Binary: macro-F1, F1 on the positive class, accuracy.
  - Multiclass: macro-F1 over the classes, accuracy.
  - Multilabel: macro/micro/per-label F1 and exact match, on the hard-majority subset.
  - Log-prob runs also save per-label mass stats; the refusal rate is filled in only when `--mass-threshold` is set.
  - The 0.5 decision rule is provisional. EXIST ICM, the all-models-answered subset and per-label thresholds are TODO.

**Check (CPU; scratch script, not committed).** Ruff passes. A tiny random-weight Llama with the cached Gemma 4 E4B tokenizer and a fake revision ran EXIST 1.3 (logprob), EDOS B (logprob) and CMSB (generate). All three wrote every file. The resume, finished-run and changed-settings guards behaved as designed. These numbers come from random weights and mean nothing.

**Issues.** On the random model, generation filled the token budget with whitespace and every output was a parse error (known xgrammar `whitespace_pattern` limit). `environment.json` is rewritten on each resume. The script hasn't been run with real weights.

**Next steps.** On Colab, run `--limit 5` per task with Gemma 4 E4B and check the Yes/No and option mass. Then do full dev runs.

## 2026-10-20 · run_inference.py helpers moved to src/utils/ (no GPU run)
**What and why.** At the user's request, moved the helpers out of `run_inference.py` into `src/utils/`. The script now keeps only the CLI, the paths, building the config and the inference loop.
- `data.py`: `load_split`. Returns the rows, the parquet path and the filters applied. Ids are cast to str. It refuses `test`.
- `repro.py`: `get_hf_token`, `resolve_revision`, `set_seeds`, `sha256_file`, `environment_info(project_root)`.
- `run_io.py`: `prepare_run_dir` (the finished-run and changed-settings guards, now raised as exceptions), `read_done`, `load_predictions`, `write_readme`.
- `metrics.py`: `answers`, `mass_summary`, `compute_metrics`. It now takes explicit `split`/`backend`/`mass_threshold` arguments instead of the argparse namespace.

Behaviour is unchanged: no logic or metric definitions changed.

**Check (CPU; same scratch smoke script as the previous entry).** Ruff passes on the script and `src/`. On the tiny random-weight model, EXIST 1.3, EDOS B and CMSB (logprob) and CMSB (generate) all ran end to end. EDOS B gave the same numbers as before the move (macro-F1 0.2, accuracy 0.667 on 3 examples; random weights, so the values mean nothing). The resume (truncated last line dropped, 2 of 3 rerun), finished-run and changed-settings guards still fire. CLAUDE.md now lists `src/utils/` and `run_inference.py`.

## 2026-10-20 · Rule: helpers go into modules (docs only, no run)
At the user's request, added `.claude/rules/code-structure.md`. Scripts and notebooks stay thin; helpers move into `src/` modules and are imported once any of these holds:
- a script or notebook has more than 3 helpers or ~100 lines of helper code;
- a helper is needed in 2+ places;
- a helper is research logic that must be the same in every run.

Updated the two lines that conflicted with the rule:
- CLAUDE.md: "Don't add more to `src/` unless asked" now points to the rule. New scripts still need a request.
- `notebooks/CLAUDE.md`: helpers copied into 3+ notebooks used to be flagged for a later refactor; they are now moved at 2 notebooks.

## 2026-10-20 · CLAUDE.md restructured via /init (docs only, no run)
Added a Commands section (ruff, `run_inference.py` smoke/resume, no tests yet) and an Architecture section (pipeline, three-place class-order invariant, refusal semantics, `HF_HOME` import order, run-folder contents, bf16 tie issue). Research rules kept unchanged; two README points added (CMSB split can't be regenerated; 02 reads test label-blind). Dropped "Later (not now)", now covered by `.claude/rules/code-structure.md`. Noted that the `requirements.lock` comment in `pyproject.toml` is stale (not fixed).

## 2026-10-20 · 15-LLM registry and label-blind test inference (no GPU run)
**Why.** The supervisor asked for 15 prompted LLMs instead of 5. The user set the size cap at 8B total params or less, 8B included (model-card figure), so models are comparable. The cap replaces two existing picks: Qwen3.5-9B with Qwen3.5-4B, and Ministral 3 8B with Ministral 3 3B. gpt-oss is excluded (gpt-oss-20b is 21B total / 3.6B active; there is no OpenAI open-weight model at 8B or less). The user also wants LLM predictions on the test sets, so encoder/ML models can be trained on train/dev later.

**Model list.** `configs/llms.yaml`, frozen 2026-10-20:
- qwen3.5-4b, gemma4-e4b, ministral3-3b, olmo3-7b, llama3.1-8b;
- phi4-mini, granite4.1-8b, nemotron3-nano-4b, apertus-8b, smollm3-3b;
- glm-edge-4b, falcon-h1-7b, exaone4-1.2b, aya-expanse-8b, granite-guardian-3.3-8b.

How the list was built and checked:
- **Research.** A research agent checked every repo on its Hugging Face model page (https://huggingface.co/<repo_id>).
- **Selection criteria.** Instruct checkpoints, fit on a 40 GB A100, released at least 2 weeks ago, family diversity, including small, fully open and safety models.
- **Pins.** The revision shas, gating and licences come from the HF API on 2026-10-20.
- **Correction.** Gemma 4 E4B is Apache-2.0 and not gated; an earlier chat note said it was gated. Gated: Llama 3.1 (manual approval), Aya Expanse (automatic approval). Non-commercial: Aya (CC-BY-NC), EXAONE.
- **Param figures.** Hub counts that differ from the cards: Granite 4.1 9B, Falcon-H1 8B, Qwen3.5-4B 5B including vision. The card figures are used.
- **Backups.** Qwen/Qwen3.5-2B, CohereLabs/tiny-aya-global, internlm/internlm3-8b-instruct, Qwen/Qwen3Guard-Gen-8B.

**Code.**
- **Registry.** `src/utils/registry.py` (`load_llm_registry`, `LLMSpec`). `run_inference.py --model <name>`; the `--model-id/--model-name/--revision` flags override it.
- **Per-model options.** `prompts.ChatOptions` (prefill, chat_kwargs) is threaded through `render`, the logprob scorers, `JsonGenerator` and `infer_single`. `load_hf_model` takes `fp8_dequantize` (Ministral 3) and `trust_remote_code`. The options are logged in `config.json`.
- **Label-blind test.** `run_inference.py --split test`:
  - `load_split` reads only `id, text_norm, split` (plus CMSB `annotated`).
  - EDOS B/C run on all test posts; the gold-sexist subset is chosen in 99.
  - No metrics; `complete.json` and a "predictions only" README are written.
  - `complete.json` now marks every finished run; older runs are recognised by `metrics.json`.
- **Rule text.** The test-set rule in CLAUDE.md, notebooks/CLAUDE.md, data/CLAUDE.md and README.md now reads: test labels only in 99; label-blind test text reads allowed.

**Check (CPU; scratch scripts, not committed).**
- `ruff check src run_inference.py` passes.
- **Loader.** On test, `load_split` returns no label columns. Rows: CMSB 1,126 annotated of 2,725; EDOS 4,000 of 4,000; EXIST 978 of 978.
- **Tiny model.** On a random-weight model:
  - test runs for CMSB, EDOS B and EXIST 1.3 wrote `predictions.jsonl`, `complete.json` and the README, with no metrics;
  - a rerun of a finished test run is refused;
  - an EDOS B dev run gives the same numbers as before (meaningless values from random weights).
- **Tokenizers** (tokenizer files only, no weights): 13/15 pass.
  - Yes/No variants and the letters A–K are single tokens, and the chat templates render.
  - Qwen3.5-4B, SmolLM3, Nemotron and EXAONE end with an empty think block when thinking is off.
  - Llama 3.1 and Aya are not checked yet: gated, and there was no HF_TOKEN locally.

**Issues.**
- **Granite Guardian is blocked** in the registry. Without `guardian_config`, its chat template judges the built-in "harm" criterion, not our label. It needs an adapter: the label definition as `custom_criteria`, the post as the user turn, then yes/no scored after `<score>`.
- **Ministral 3 FP8 dequantization** on transformers 5.20.1 is untested on GPU.
- **EXIST config rename.** The working copy of `configs/data_prep.yaml` renames EXIST `task_1_3` to `task_1_2` but keeps the five 1.3 categories. `check_against_config` now fails for every run. This was not changed here; it was flagged to the user.
- **bf16 tie issue** still pending.

**Next steps.**
- Decide on the data_prep.yaml rename.
- Run the tokenizer check for Llama and Aya with HF_TOKEN.
- Build the Guardian adapter.
- Run GPU smoke tests (`--limit 5`) per model.

## 2026-10-20 · EXIST Task 1.2 added; task_1_3 name restored (data run)
**Why.** The user asked to add EXIST Task 1.2 (source intention) next to 1.1 and 1.3. They had also renamed the 1.3 config key to `task_1_2`, which broke the schema/config check for every run. Keys follow the official EXIST numbering, so `task_1_3` is restored.

**Task definition.**
- `task_1_2` is multiclass on column `intention`, with classes `["NO", REPORTED, JUDGEMENTAL, DIRECT]`. These are the official 4 classes in the order of the EXIST validator's `LIST_LABELS_SUBTASK2`.
- NO is a class, so 1.2 runs on all tweets, like 1.3. Metrics score the hard-majority subset.
- Prompt definitions come from `EXIST2025_Annotation_guidelines_Tweets.pdf`: question 2 for the three intention classes, question 1's sexism definition for NO. The text was extracted with macOS PDFKit into the scratchpad; nothing was installed.
- `"NO"` must be quoted in YAML: the registry check caught a bare `NO` parsed as `False`.

**Code.**
- `configs/data_prep.yaml`: `task_1_2` added, `task_1_3` restored.
- `src/schemas/tasks.py`: `ExistTask12`.
- `src/prompts/templates/exist/task_1_2.yaml`: lettered options A–D.
- **EDOS-only filter.** `load_split` keeps the gold-sexist filter for EDOS B/C only. The multiclass metrics drop rows with null gold, and the `subset` string now depends on the dataset.
- **01_data_prep:**
  - reads the 1.2 hard/soft golds;
  - builds `intention`, `intention_soft`, `intention_votes`, `intention_n_annot` (`-` counts as NO, UNKNOWN is dropped);
  - asserts that the votes reproduce the soft gold, that 1.2 NO agrees with 1.1 NO, and the hard counts (all languages: 5,496 train, 841 dev).

**Run.** `20261020-1100_all_prep_none_dataprep_s42`: 01 run locally as a plain-Python cell runner from `notebooks/` (no nbconvert installed), with `update_manifest: true`.
- All assertions passed.
- Only `exist_{train,dev,test}` changed; the CMSB and EDOS processed content is unchanged.
- `results/data_manifest.json` was rewritten, with 4 new raw entries (the 1.2 golds).
- English 1.2 hard counts:

  | split | NO | REPORTED | JUDGEMENTAL | DIRECT |
  |---|---|---|---|---|
  | train | 1,733 | 194 | 148 | 545 |
  | dev | 250 | 35 | 28 | 87 |

  Source: the run's `metrics.json`.

**Check.** Tiny-model smoke test of `--dataset exist --task task_1_2`. Dev scores the hard-majority subset; test is label-blind and writes predictions only. Ruff passes.

**Open.**
- The official 1.2 evaluation is hierarchical ICM / ICM-soft; we compute macro-F1 only (TODO).
- Position bias over the option order is not handled.
- The 01 notebook still holds cell outputs from the user's last run; strip them before committing.

## 2026-10-20 · notebooks/20_llm_runs.ipynb: batch orchestration (no GPU run)
**Why.** The user wants the LLM runs driven from a notebook on Colab.

**What.**
- **`20_llm_runs`** builds the jobs from `CONFIG`: models (all 15 by default) × the 7 tasks (CMSB binary, EDOS A/B/C, EXIST 1.1/1.2/1.3) × splits (dev, test). It runs each job as a `run_inference.py` subprocess, which frees GPU memory and isolates crashes, and logs each job's output to a file in the batch folder.
- **Resume.** Run names come from a fixed `CONFIG["date"]`, so a rerun skips `done` jobs and resumes `partial` ones.
- **Statuses.** A job is blocked, done, partial or todo.
- **Safety.** `dry_run: True` is the default, so "run all" never launches GPU work by accident.
- **Colab layout.** The processed data is copied to `/content/data`. HF weights go to local `/content/hf_cache`: Drive is slow and too small for 15 models. Outputs stay on Drive.
- **Records.** Each session writes `status_*.csv` and `failed_*.json` into `OUTPUT_ROOT/{date}_all_all_llms_batch_s42/`, and never overwrites earlier ones.
- **Helpers.** `src/utils/jobs.py` holds `build_jobs`, `Job`, `job_status`, `status_table`, `run_job`, `tail`, `method_name` and `make_run_name`. `run_inference.py` now builds its run names with the same helpers.

**Check (CPU).**
- Ruff passes.
- The notebook runs top to bottom as a plain-Python cell runner in dry-run mode: 210 jobs, 196 todo, 14 blocked (Granite Guardian). The test batch folder was deleted afterwards.
- The launch path was tested with a stand-in script that loads a tiny random model:
  - 4 jobs finished (CMSB binary and EXIST 1.2, dev and test, `--limit 2`) and Guardian showed as blocked;
  - a job with a truncated `predictions.jsonl` showed as partial and resumed to done;
  - an unknown task gave exit 1 with the error in the log tail.

**Open.**
- Think-on runs are disabled (`thinking_models: []`). JSON-constrained decoding starts at the first generated token, so it suppresses the thinking; a generate-then-score path is needed first.
- I can't reach the Colab runtime from here: the IDE kernel tool found no open notebook.
- No GPU job has been run yet.
