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

## 2026-10-20 · Parallel batch runs; test split only (no GPU run yet)
**Parallel jobs.**
- `src/utils/jobs.py` gains `run_jobs_parallel`. It runs several `run_inference.py` processes on one GPU at once. Jobs start in order while the sum of their `vram_gb` estimates fits `gpu_budget_gb`, up to `max_parallel`; a job bigger than the budget runs alone.
- Each job stays batch-1 in its own process, so its scores don't depend on what runs beside it. Interrupting the cell terminates the children and leaves their runs resumable.
- **Memory estimates.** `vram_gb` per model in `configs/llms.yaml` is the Hub safetensors size at the pinned revision plus 3 GB, rounded up. Ministral 3 is FP8: its repo holds two 4.67 GB copies, of which one is loaded, dequantized to about 9.3 GB in bf16. `run_inference.py` now records the measured `peak_gpu_gb` in `complete.json`, so the estimates can be checked.
- **Notebook defaults.** `max_parallel: 4`, `gpu_budget_gb: 70` (the Colab runtime is an A100-SXM4-80GB).
- **Check (CPU, tiny-model stand-in):**
  - 6 jobs with a 40 GB budget ran two 19 GB jobs at once, and the small jobs filled in afterwards; all finished.
  - A SIGINT stopped the children with no orphans. The interrupted job showed as `partial`.
  - The first two interrupt tests were invalid: the jobs finished too fast, and then a background shell ignored SIGINT. Both are recorded here and were rerun correctly.

**Test only.** The user decided on 2026-10-20 to run the LLMs on test only, with no dev runs, because evaluation is on test. Consequences:
- Per-label thresholds and the soft-refusal mass cutoff can no longer be tuned on dev. They must be fixed a priori and recorded before any test scoring. Proposed: P(Yes)/mass >= 0.5 for binary/multilabel, argmax for multiclass, and a soft-refusal cutoff TODO, to be confirmed by the user.
- EXIST test is unlabelled, so EXIST test scores come only from the official EvALL submission.
- The first CMSB dev run (Gemma) remains the only dev result.

**Colab runtime.** The kernel holds numpy 2.1.3 in memory, while numpy 2.5.4 is installed from the pins, which causes the `_blas_supports_fpe` errors. A session restart is needed.

**Security.** An HF token had been pasted into the 20 notebook's secrets cell. The cell was reverted to `userdata.get("HF_TOKEN")`, and the user was asked to revoke the token.

## 2026-10-20 · Granite Guardian replaced by AI21 Jamba2-3B
**Why.** Granite Guardian 3.3-8B is a guard model. Without `guardian_config`, its chat template judges its built-in "harm" criterion instead of our label, and building an adapter would give it a prompt format no other model gets. It was blocked and never run. The user decided to drop it and replace it with one general-purpose instruct model.

**Selection.** A research agent checked candidates against the existing rules: open weights, instruct model, <= 8B total params (model-card figure), a family not yet in the list, a plain chat template whose first reply token can be Yes/No, stock transformers, and a licence allowing research use. The user chose the top pick.
- **Chosen: `ai21labs/AI21-Jamba2-3B`** at `525c6c8e1d9f5bddedfbdc1dbb0ade2df84230c9`.
  - 3B params (card: "At just 3B parameters"; Hub safetensors 3.03B), Apache-2.0, not gated. The sha and licence were rechecked against the Hub API on 2026-10-20.
  - ChatML template with no thinking and no default system prompt; no `prefill` or `chat_kwargs` needed. `vram_gb` 10 (6.39 GB + 3).
  - New family: a hybrid Transformer–Mamba model from AI21.
- **Runner-ups:**
  - `LiquidAI/LFM2.5-1.2B-Instruct`: very small, and EXAONE already covers 1.2B.
  - `tencent/Hunyuan-7B-Instruct`: its licence excludes the EU, UK and South Korea, and thinking is on by default with an untested workaround.
- **Rejected:** InternLM3-8B and MiniCPM4.1-8B (over 8B, remote code); DeepSeek-R1-Distill-7B and Jamba-Reasoning-3B (forced reasoning); Mistral-7B v0.3 (family already present).

**Note.** `mamba-ssm` and `causal-conv1d` are not installed (no new packages without approval), so transformers uses its PyTorch Mamba path. It may be slower and can differ slightly in numbers from the CUDA kernels; `environment.json` (pip freeze) records which packages were present.

**Changes.** `configs/llms.yaml` (Guardian block replaced) and `CLAUDE.md`. The model count stays 15. No run of Jamba yet; its 7 test jobs run in the next `20_llm_runs` session with the same `date` (20261020).

## 2026-10-20 · First full test batch (`20261020_all_all_llms_batch_s42`)
**What.** `notebooks/20_llm_runs.ipynb` ran every (model, task) job on test with the logprob backend, seed 42, `max_parallel` 7 and `gpu_budget_gb` 70 on an A100-80GB. The session still used the old registry, with Guardian blocked and Jamba not yet added.

**Outcome** (from the notebook's printed log; per-session tables in the batch folder's `status_*.csv` / `failed_*.json`):
- **91 of 98 jobs exited 0.** 13 models finished all 7 tasks: Qwen3.5-4B, Ministral-3B, OLMo-3-7B, Phi-4-mini, Gemma-4-E4B, Nemotron-3-Nano-4B, SmolLM3-3B, GLM-Edge-4B, EXAONE-4-1.2B, Llama-3.1-8B, Granite-4.1-8B, Apertus-8B and Falcon-H1-7B.
- **All 7 Aya Expanse 8B jobs failed** (exit 1 after 0.3 min) with `403 Client Error ... Cannot access gated repo` for `CohereLabs/aya-expanse-8b`. The HF token's account hasn't been granted access. This is an access problem, not a model or code failure; no Aya predictions were produced.
- Granite Guardian's 7 jobs were blocked and not run (replaced by Jamba2-3B, entry above).
- Wall-clock job times ranged from 1.2 min (CMSB, Qwen) to 21.7 min (EDOS C, Nemotron), with up to 7 jobs sharing the GPU. Falcon-H1 (hybrid Mamba, no `mamba-ssm`) was slower than the other ~8B models, e.g. 13.6 min on EDOS A vs 6.0–8.0 min.
- Test runs are label-blind; no metrics yet (scored in `99_final_eval`).

**Next.**
- Accept the Aya Expanse licence on the Hub with the token's account.
- Sync the updated `configs/llms.yaml` to Drive.
- Rerun `20_llm_runs` with `date` 20261020. Only Aya (7) and Jamba2-3B (7) should run.

## 2026-10-20 · SHAP / LIME attribution helpers (no runs yet)
**What.** Added `src/utils/explain.py`, re-exported from `src.utils`:
- `LabelScorer`: maps posts to log P(Yes)/P(No) for one label, using the same prompt, `render` and `next_token_probs` as `score_yes_no`. It runs one post per forward pass so scores match the main runs. Posts with zero Yes or No probability return NaN and are counted (`n_nan`), not filled.
- `shap_explain`: SHAP Partition values per word on the log-odds scale, with a `\W+` word split and `...` as the mask. Only the post is masked, never the instructions.
- `lime_explain`: LIME weights for P(Yes) with the same word split and a fixed `random_state`. It raises an error if any perturbed post is unscorable.
- Binary/multilabel (Yes/No) tasks only; EDOS B/C (lettered options) are not covered yet.

**Why.** The user asked how to get SHAP/LIME values. These are post-hoc: they need new forward passes on perturbed posts, so they can't be computed from saved `predictions.jsonl`. The existing runs don't need rerunning.

**Packages.** Pinned `shap==0.54.1` and `lime==0.2.0.1` (the versions in the local `.venv`) in `requirements.txt` and the 00 install cell.

**Check.** A CPU smoke test used a fake tokenizer/model whose log-odds equal a sum of word weights (alpha 2.0, beta -1.0, gamma 0.5, delta 0). SHAP recovered the weights exactly, with base 0 and additivity holding. LIME ranked the words in the same order (R² 0.98) and gave identical results on a rerun with seed 42. No real model was run.

**Caveats.**
- Cost is n_posts × n_labels × evals forward passes, so use a small, seeded dev sample.
- The bf16 score-grid issue may give zero attributions for small perturbations.
- SHAP values depend on `max_evals`; LIME depends on the seed.

**Next.**
- Pick the models and the dev sample size.
- Run on dev only (never test), saving to `OUTPUT_ROOT/{run_name}/`.
- Add a Captum IG cross-check once the encoders exist.

## 2026-10-20 · Final test-evaluation protocol, frozen before any test label is read
**Why.** The 105 test runs of `20261020_all_all_llms_batch_s42` (15 models × 7 tasks; folders in `outputs/final_runs_result/outputs/`) are finished. Every run has `mass_threshold: null`, so no soft refusal was flagged at inference. The user chose these rules on 2026-10-20. They are recorded here before `notebooks/99_final_eval.ipynb` reads any label column, and they must not change after test scores are seen.

**Pre-scoring audit** (label-free; `predictions.jsonl` + `config.json` only):
- All 105 runs have `complete.json`, and `n_pred` equals `n_examples` (CMSB 1126, EDOS 4000, EXIST-en 978).
- No duplicate IDs. One prompt sha per run. `revision` equals `revision_requested`.
- Answer mass is low on several lettered-option tasks: Aya EDOS C p05 0.000, with top tokens "The"/"Based"; Gemma EDOS C p05 0.10; Granite EDOS C p05 0.07; Granite EXIST 1.2 p05 0.20.

**Frozen rules.**
- **Soft refusal:** a label is refused when its answer mass (P(Yes)+P(No), or the summed option-letter mass) is < **0.5**. This value is chosen a priori, since no dev runs exist. Sensitivity grid {none, 0.1, 0.5, 0.9}: reported next to the primary number, never used to pick it.
- **Decisions:** binary/multilabel P(Yes)/mass ≥ 0.5; multiclass argmax over options. A priori, no dev tuning.
- **Strict scoring** (`compute_metrics`): a refused label is scored wrong.
- **Fair subset:** posts on which all 15 models answered every label of the task at the 0.5 cutoff.
- **Metrics:**
  - CMSB / EDOS A: macro-F1 (primary), positive-class F1, accuracy.
  - EDOS B / C: macro-F1 (primary) and accuracy on gold-sexist test posts (the SemEval-2023 Task 10 B/C setting). Predictions on non-sexist posts are ignored.
- **EXIST:** test is unlabelled; scores come from EvALL only. Locally we report refusals and inter-model agreement only.
  - Hard and soft submission files are written per model. A refused item is left out of the hard file.
  - 1.3 soft `NO` = Π(1 − P(Yes)) over the 5 labels (independence assumption).
- **Uncertainty:** 95% percentile bootstrap CIs (B = 10,000, seed 42, resampling posts).
  - Each model is compared with the task's best model by a paired bootstrap of the macro-F1 difference.
  - Binary tasks also get McNemar (exact).
  - Holm correction across the 14 comparisons per task, α = 0.05.
- **CMSB leak rows:** the 33 test rows in a `leak_group` count in the main number, with a second row that excludes them.
- **Outputs:** `results/final_eval/{tables,figures}` and `summary.json`; EvALL files in `results/evall_submissions/`.

**Known deviations to report.**
- No dev tuning of thresholds or of the refusal cutoff.
- Qwen3.5-4B thinking-on was not run (`thinking_models: []`).
- EXIST is English only and pending EvALL. The official validator expects 2,076 (es + en) items.

## 2026-10-20 · Final test evaluation of the 15 LLMs (`20261020_all_all_llms_finaleval_s42`)
**What.** Scored the 105 test runs under the protocol frozen in the entry above.
- New notebook `notebooks/99_final_eval.ipynb`.
- New helpers:
  - `src/utils/metrics.py`: `apply_mass_threshold`, `refusal_rates`, `answered_ids`, `strict_arrays`.
  - `src/utils/stats.py` (new): bootstrap, McNemar, Holm, Fleiss' κ, ECE.
  - `src/utils/evall.py` (new): EXIST submission export.
  - `src/utils/report.py` (new): tables and figures.
  - `src/utils/run_io.py`: `load_finished_runs`.
  - All re-exported from `src.utils`.

**Checks.**
- `compute_metrics` on the CMSB Qwen dev run reproduces its `metrics.json` (macro-F1 0.80229…).
- `strict_arrays` + `macro_f1_counts` reproduce `compute_metrics` for every test model and task (asserted in 99), and equal sklearn on random 4- and 11-class data with refusal codes.
- `apply_mass_threshold(·, None)` is the identity.
- Same seed gives identical bootstrap results.
- Fleiss' κ gives 0.210 on the Wikipedia worked example.
- The notebook was run locally top to bottom; `.venv`, CPU, code cells executed as a script, since nbconvert is not installed.

**Results.** Sources: `results/final_eval/tables/` and `figures/`. Full summary in the notebook's last cell and `results/final_eval/summary.json`. Strict macro-F1, test, t = 0.5:

| Setting | Best model, macro-F1 [95% CI] | Not separable after Holm |
|---|---|---|
| CMSB | Granite 4.1-8B, 0.821 [0.797, 0.843] | 5 models |
| EDOS A | Gemma-4 E4B, 0.712 [0.696, 0.726] | 2 models |
| EDOS B | Falcon-H1-7B, 0.508 [0.466, 0.549] | none |
| EDOS C | Gemma-4 E4B, 0.256 [0.227, 0.284] | Falcon-H1-7B |

- Soft refusals occur only on lettered-option tasks:
  - EDOS C: Aya 99%, SmolLM3 98%, Granite 83%, Gemma 13%, Ministral 10% of posts.
  - EXIST 1.2: Granite 17%.
- The EDOS C fair subset is empty; EDOS B has 865 of 970 posts.
- Fleiss' κ across models: 0.57 (CMSB) down to 0.05 (EDOS C).
- Every model over-predicts sexism on EDOS A.

**EvALL.**
- 90 files (15 models × 3 tasks × hard/soft) are in `outputs/20261020_all_all_llms_finaleval_s42/evall_submissions/`. The manifest with sha256 is also in `results/final_eval/tables/evall_manifest.csv`.
- They pass the official JSON schema. The only validator complaints are the count checks: English only, 978 of the 2,076 items it expects.
- **Deviation from the protocol entry:** the files went to the run folder rather than `results/evall_submissions/`, because they total about 13 MB. Only their location changed.

**Issues / caveats.**
- Thresholds and the refusal cutoff were fixed a priori, with no dev runs.
- One seed and one prompt per task.
- Qwen3.5-4B thinking-on was not run.
- scipy and matplotlib are used but not pinned.
- The batch README was stale; a correction note was appended to it, with no other change to the run folder.

**Next.**
- Confirm the EvALL team name and whether English-only runs are accepted, then submit and add the EXIST scores.
- Decide on the Qwen thinking run.
- Pin scipy/matplotlib (needs approval).

## 2026-10-20 · LLM-baseline results report (LaTeX → PDF)
**What.**
- `docs/report/llm_baselines_report.tex` with `refs.bib`, compiled to `docs/report/llm_baselines_report.pdf` (12 pages).
- It covers data, models and prompting, the frozen protocol, results, EXIST status, limitations and reproducibility, for the run `20261020_all_all_llms_finaleval_s42`.
- HEFT is not covered.

**How the numbers stay tied to the run.**
- Tables are generated, not typed. The new section 11 of `99_final_eval` writes `results/final_eval/paper/{main,secondary,refusal,sensitivity,calibration,agreement,models}.tex` with the new `write_paper_table` helper in `src/utils/report.py`.
- The report `\input`s those tables and the figures in `results/final_eval/figures/`.
- Numbers in the prose were checked against the CSVs. Two rounding mismatches (EDOS A gold rate 0.242, max predicted rate 0.760) and one overstatement (scores at the 0.9 cutoff: only SmolLM3, Ministral, Apertus, Aya and Granite collapse) were corrected in the report and in the notebook summary.

**Citations.**
- Samory et al. 2021: checked against arXiv and the GESIS record.
- Kirk et al. 2023: checked against the ACL Anthology.
- Plaza et al. 2025: authors, CEUR volume and pages come from search records. The exact title is still TODO (marked in `refs.bib`).
- The statistics references are cited from standard sources.

**Tooling.** Tectonic 0.17.0 installed via Homebrew (user-approved) to compile the PDF: `cd docs/report && tectonic -X compile llm_baselines_report.tex`.

## 2026-10-20 · Report revision: motivation, conclusion, gold-label explanation, no HEFT
**What changed in `docs/report/llm_baselines_report.tex`.**
- Removed every HEFT mention (user request).
- Added a motivation section and a conclusion section. The conclusion has 5 points and the protocol contribution, each tied to numbers already in the tables.
- Added an explanation of gold labels (hard/soft) and of why the EXIST test gold is withheld.
- Rewrote "Limitations" as "Scope and design choices" (user request). The same facts are kept: no dev calibration, a single prompt and seed, format failures on EDOS C, thinking disabled, English only and EXIST pending. Only the framing changed.
- Two claims were softened to match the evidence. Dev calibration "may improve" the results, and there is "no consistent relation" between size and performance (EDOS B ρ = 0.52, p = .045 uncorrected).

**EXIST gold status.** A web search on 2026-10-20 found no public release of the EXIST 2025 test gold. EXIST 2025 is listed in the EvALL 2.0 task repository (evall.uned.es), so post-campaign evaluation through EvALL appears possible. This is not yet tested by a submission.

## 2026-10-20 · Four-reviewer audit of the report; corrections; new report
**What.** Four review agents checked the report: a numbers audit, a methods and claims review, a writing review, and a citation and related-work check. From their findings a new report was drafted, `docs/report/zero_shot_sexism_llms.{tex,pdf}` (18 pages). It has no repo-internal references and no HEFT mention. The earlier `llm_baselines_report.*` is kept unchanged for comparison.

**Corrections to earlier results** (all in `99_final_eval`; earlier log entries are left as written):
- **Leak rows.** 33 CMSB test rows carry a `leak_group`, but only **3** are among the 1,126 scored annotated posts. The "33" in earlier text referred to the whole test file. `macro_f1_no_leak` was already computed on the scored posts, so its values are unchanged.
- **Tie marks used the wrong test.** For binary tasks, "not significantly different from the best" had been decided with McNemar, which tests accuracy. The frozen protocol names the paired bootstrap of macro-F1 as the comparison, with McNemar as an addition. The code now follows the protocol: bootstrap for every task, McNemar reported as secondary in `paper/significance.tex`. Effect: CMSB is unchanged, and on EDOS A Qwen3.5-4B is now tied with Gemma-4-E4B (Holm p = 0.056). This applies the declared rule; no threshold or metric changed.
- **Refusal denominators.** EDOS B/C refusal rates in the paper are now over the 970 scored sexist posts; the all-post rate is kept as a separate column. Gemma-4-E4B EDOS C: 13.5% of all posts, 0.6% of scored posts.
- **"No exact 0.5 scores" was wrong.** One EXIST 1.3 label of Gemma-4-E4B is exactly 0.5; it is predicted positive under ≥ 0.5. The labelled tasks have none.
- **Sensitivity wording.** At τ = 0.9, Granite also drops on EDOS B (0.362 → 0.271). On the binary tasks only Ministral changes (CMSB 0.726 → 0.714, EDOS A 0.502 → 0.454).

**Additions.**
- `display_name` in `configs/llms.yaml` and `LLMSpec`. It is used only in figures and tables and never reaches run configs.
- A signed calibration gap per model.
- A generated prompt appendix (`paper/prompts.tex`).
- EvALL no-cutoff file variants, because EvALL's handling of missing items is unconfirmed; 180 files in total.
- `git diff d054aee d758689 -- src run_inference.py` is empty, so all 105 runs used the same inference code.

**Report content decisions** (user, 2026-10-20):
- Dev runs are not part of this study: train and dev are reserved for the supervised models, and the zero-shot study uses test only. The report states this in the data section.
- HEFT is removed.
- Limitations are written constructively without dropping any fact.
- Related work and EDOS positioning numbers were verified from the sources: Kirk et al. 2023 Table 3, and Riahi Samani et al. 2025 Table 1 (test columns).
- The EXIST 2025 overview title was confirmed from the PDF. The bib was corrected and extended.

## 2026-10-23 · Full pipeline documentation in docs/ (docs only, no run)
**What and why.** The user asked for complete documentation, from data creation to the final evaluation. Added `docs/README.md` (index, pipeline diagram, status, glossary) and nine pages: `01-setup`, `02-data-pipeline`, `03-data-analysis`, `04-llm-inference`, `05-batch-runs`, `06-final-evaluation`, `07-reproducibility`, `08-code-reference`, `09-known-issues`. Written from the code, configs, notebooks, run folders and this log; no code or data changed. The `docs/` line in `CLAUDE.md` and `README.md` now points to them.

**Checks.** Headline numbers in `06-final-evaluation.md` were re-read from `results/final_eval/tables/main_results_full.csv` (best model, CI, median, majority, ties per task). Other numbers are quoted from the run files named next to them.

**Found while writing** (listed in `09-known-issues.md`, not fixed):
- `notebooks/20_llm_runs.ipynb` (working copy) again holds a hard-coded HF token instead of `userdata.get("HF_TOKEN")`.
- `configs/data_prep.yaml` has `update_manifest: true`.
- `predictions.jsonl` is tracked at the repo root; `predictions (1).jsonl` is untracked there.
- Stale text: README lists 99 as planned; `data/README.md` and 01's intro omit EXIST 1.2; 01's Findings cell still reports the split file missing; 00 says Gemma is gated.
- The test batch's Colab runtime was Python 3.13.15, outside `pyproject.toml`'s `<3.13`.
- Run folders `20261019-1557_all_prep_none_dataprep_s42` and `20261019-1619_all_analysis_none_eda_s42` have no log entry.

## 2026-10-23 · What the soft refusals are (label-free check, no run)
**What and why.** The user asked what the refusals are. Tallied the top-1 next token (`top_k[0]`, stored per prediction) on posts with answer mass < 0.5, from the test run folders in `outputs/final_runs_result/outputs/`. No labels read; no file written.

**Result** (refused posts / all 4,000 EDOS or 978 EXIST posts; top token counts):
- EDOS C Aya-Expanse-8B 3,944: "The" 3,667, "Based" 273. SmolLM3-3B 3,932: "The" 3,902. Granite-4.1-8B 3,319: "The" 3,314.
- EDOS C Gemma-4-E4B 539: "None" 506. Ministral-3-3B 392: "None" 240, "**" 139.
- EDOS B SmolLM3-3B 215: "The" 205. EXIST 1.2 Granite 164: "To" 90, "The" 62.
- No refusal-like first token ("I", "Sorry") among the top tokens above.

**Interpretation.** The "soft refusals" are mostly format non-answers (the model starts an explanation, or rejects every option with "None"), not safety refusals. Wording in the paper should say "non-answer" (as `refusal.tex` does). Only the first token is seen, so what the model would have said next is unknown.

## 2026-10-23 · Few-shot prompts from train (code + frozen shot sets, no run)
**What and why.** The user asked for few-shot prompts built from training examples. User decisions: seeded, stratified selection (no hand-picking); all 7 tasks; 1 shot per class. A second motive: the soft refusals in the zero-shot test runs are mostly format non-answers (entry above), and shot turns show the expected answer format.

**Code.**
- New `src/prompts/fewshot.py`: `select_shots`, `freeze_shots`, `load_shots`.
- `PromptTemplate.yes_no_messages` / `option_messages` take `shots`. Each shot becomes a user turn (same template) plus an assistant turn (`Yes`/`No`, or the letter).
- `shots` is threaded through `score_example` and `infer_single`.
- `run_inference.py` gains `--shots K --shot-seed S`. It is log-prob only, refuses shot IDs that are in the evaluated split, and adds `shots_file`, `shots_sha256` and `shot_ids` to `config.json`.
- Run names use `logprob-fs{K}`. `Job`/`build_jobs` take `shots`.
- `20_llm_runs.ipynb` has `CONFIG["shots"]` (default 0) and a freeze cell.

**Selection rules** (in code, stored in each file):
- Train only.
- Excluded: rows with `leak_group` or `dup_group`, `empty_norm` rows, rows without a hard label, and rows with more than 50 words. CMSB: annotated originals. EXIST: `en`. EDOS B/C: gold-sexist rows.
- Binary: 1 Yes + 1 No. Multiclass: 1 per class.
- EXIST 1.3: per label, 1 Yes + 1 No, where the No is a sexist tweet without that label.
- Seeded shuffle (seed 42).

**Frozen files** (IDs and answers only, no text): `configs/fewshot/cmsb/binary_k1_s42.yaml`, `configs/fewshot/edos/task_{a,b,c}_k1_s42.yaml`, `configs/fewshot/exist/task_1_{1,2,3}_k1_s42.yaml`. They are frozen before any few-shot run, and the same shots are used for every model.

**Checks** (local, no GPU):
- `ruff check` passes.
- Zero-shot messages and template sha256 are unchanged on all 7 tasks (compared with the pre-change output). Zero-shot run names are unchanged.
- Every shot ID is in train and in neither dev nor test, and none has a `leak_group`. Every stored answer matches the train gold.
- Measured prompt length (Gemma-4-E4B tokenizer, first dev post, longest prompt; zero-shot → 1-shot): CMSB 168 → 503; EDOS A 95 → 305, B 149 → 773, C 358 → 4,300; EXIST 1.1 193 → 503, 1.2 232 → 973, 1.3 188 → 453.

**Deviations / notes.**
- With k=1 and the same seed, the three binary sets all come out in Yes-then-No order, so the last shot before the question is always a No. Watch for a No bias.
- EDOS C prompts are about 12× longer, because the option list is repeated in every shot.
- Not done: generate-backend shots, multiple shot seeds.

**Next.**
- GPU runs (`CONFIG["shots"] = 1`), not started.
- The 2026-10-20 test-only decision means few-shot vs zero-shot is compared on test. The shots were fixed before any few-shot result.

## 2026-10-23 · Evaluation protocol review (docs only, no run)
**What and why.** The user asked whether the current evaluation is the best way to evaluate these tasks, and to research better options. The frozen protocol (`docs/06-final-evaluation.md`) was reviewed against the literature, with web searches in this session. Result: `docs/10-evaluation-review.md`, linked from `docs/README.md`. No code, config or notebook changed, nothing was run, and no test labels were read (only test parquet column names, to confirm the soft gold columns exist).

**Conclusions** (details and sources in `10`):
- **Keep:**
  - EDOS macro-F1 with B/C on gold-sexist posts matches the official setting (Kirk et al. 2023). This closes the open B/C scope question for the main setting.
  - EXIST ICM / ICM-soft via EvALL.
  - Paired bootstrap + Holm, and strict refusals.
- **Main gaps:**
  - (1) One prompt per task, so the CIs ignore prompt variance, which can reorder models (Sclar et al. 2024; Mizrahi et al. 2024).
  - (2) A fixed 0.5 threshold on Yes-biased scores (14/15 models over-predict on CMSB, all 15 on EDOS A, `results/final_eval/tables/positive_rate.csv`). Proposed: add AUROC/AUPRC, label-free calibration rules, and possibly dev thresholds.
  - (3) No soft-label evaluation for CMSB/EDOS, although `sexist_soft`, `category_soft` and `vector_soft` are in the test files.
- **Smaller gaps:** risk–coverage / AURC and pairwise fair subsets (the all-model fair subset is empty for EDOS C); tie wording; Brier score; CMSB original vs adversarial-modification analysis; a contamination probe; EDOS C per-class n.

**Integrity.** Test labels were already read on 2026-10-20, so every addition is post-hoc: secondary, never replacing the primary numbers. Multi-prompt runs and dev thresholds need a new protocol entry before scoring. Dev thresholds would reverse the 2026-10-20 test-only decision (user's call).

**Open (TODO verify):** EXIST ICM / ICM-soft definitions and the hierarchy use (the overview PDF could not be machine-read here), and the metric Samory et al. 2021 report for CMSB. Specific literature numbers come from abstracts and snippets; check them in the full text before citing.

**Next.** The user chooses which items to implement. CPU-only items 1–4 of the suggested order use the existing predictions.

## 2026-10-23 · 10-agent research sweep → OSNEM roadmap (docs only, no run)
**What and why.** The user asked for 10 agents to find gaps in the evaluation metrics and inference methods, research fixes, and suggest changes for a top-tier paper. Target: the Elsevier journal OSNEM, mid-November, deadline-feasible first (user, 2026-10-23).
- Nine agents worked in parallel: eval-metrics, inference-scoring, prompting-fewshot, calibration-decisions, disagreement-soft, statistics-rigor, sota-positioning, robustness-generalisation, osnem-venue. A tenth, a reviewer-2 critic, then audited, deduplicated and spot-checked 8 key citations.
- They coordinated through a shared brief, a message board and per-agent progress logs in the session scratchpad (`scratchpad/research/`, temporary).
- Result: `docs/11-osnem-roadmap.md`, linked from `docs/README.md`. `docs/10` got a pointer to its corrections.
- No code, config, notebook or data changed; nothing was run. `git status` was identical before and after the sweep.

**Main conclusions** (details and sources in `11`):
- The frozen hard-label protocol is sound. As drafted, the paper is "major revision at best".
- **Errors found** (`11` §B):
  - Two "not separable" daggers sit next to Δ intervals that exclude 0 (CMSB Jamba2-3B, EDOS A Qwen3.5-4B). Checked against `results/final_eval/tables/main_results_full.csv`.
  - The EvALL "strict" files are lenient: omitted items score better than wrong ones.
  - English-only EvALL scoring against the bilingual gold is invalid as planned.
  - The EXIST 1.3 soft NO is computed wrongly.
  - EDOS C non-answers are first-token format artefacts; the template prefill is never applied in log-prob scoring.
  - Letter ties go to the earlier class (534 of 60,000 EDOS C predictions).
  - CMSB test includes survey-scale items and adversarial edits.
  - The EXIST 2025 tweets are probably the EXIST 2023 corpus (unconfirmed).
  - HEFT and thinking-mode doc inconsistencies.
  - The binary few-shot sets all end with a No shot.
- **Corrections to `docs/10`:**
  - Use L1/Brier, not cross-entropy, as the primary soft metric.
  - Use prior-aware calibration; plain calibration targets uniform classes.
  - The contamination probe is redesigned.
  - Prompts × posts are crossed, so the bootstrap must be two-way.
- **Decisions D1–D8 for the user** (HEFT scope, dev runs, PyEvALL, EXIST fallback, GPU budget, encoder baseline, data deposit, no model-list changes). Must-do work is about 12–13 person-days and about 4 A100 h, plus about 20 GPU h for the prompt study.

**Protocol deviation.** Agent 09 read the raw CMSB `of_id` for test items to count the test composition (451 adversarial edits among the 1,126 scored items, per-source counts). The brief forbade this because `of_id` reveals labels. Only aggregate counts were produced; no metric was computed. The frozen 2026-10-20 results are unaffected. Any CMSB contrast-set analysis must be pre-specified before it runs in `99`, and the paper must say the composition was inspected after primary scoring.

**Other notes.**
- Agent 01 ran PyEvALL v0.2.11 from the scratchpad, with two dependencies stubbed, on EXIST dev gold and synthetic predictions only. Adding the package is decision D3.
- Several literature numbers are snippet-only; check them in the full text before citing.

**Next.** The user makes decisions D1–D7, then P0 (the protocol addendum) is committed before any new analysis.

## 2026-10-23 · Addendum A1 implemented: experiments, sample mode, failure analysis, explainability (code only, no run)
**What and why.** The user asked to implement the roadmap (`docs/11`), keep EXIST, and add two things to make the paper a strong sexism-detection study: an account of why models fail to say Yes, and an AI-explainability section. They also asked for a sample flag to sanity-check background notebook runs, and for experiments tracked in dated folders on Drive.
- **Decisions** (user, 2026-10-23):
  - D1: standalone zero-shot paper; HEFT moves to a follow-up.
  - D2: label-blind dev inference allowed for pre-registered secondary analyses.
  - GPU scope: format check, answer prefix, option order, 5-prompt study, encoders.
  - EXIST scored on its labelled English dev split (the user first chose to drop EXIST, then reversed that).
- **Rules first:** `docs/12-protocol-addendum.md` fixes every new metric, rule, word list (`configs/lexicons.yaml`) and prompt variant (`configs/prompt_variants.lock`, 34 files) before anything runs. **The real post-hoc run and the new batches must wait until the user commits these three files.**

**Built.**
- **Experiment folders** `OUTPUT_ROOT/experiments/{YYYYMMDD-HHMMSS}_experiment_{name}/`, holding `experiment.json`, `runs/`, a `manifest.csv` rebuilt from the folders, `logs/` and `heartbeat.json` (`src/utils/experiments.py`).
  - **Sample (sanity) mode** in `run_inference.py` (`--sample N`) and in notebooks 20/30/40. Sample runs go only into `-sample` experiments; both `run_inference` and `99` refuse to mix them with full runs.
  - The frozen 2026-10-20 batch is registered read-only as `20261020-000000_experiment_zeroshot-frozen` the first time `99`'s post-hoc section runs.
- **`run_inference.py`:**
  - new flags `--experiment-dir`, `--sample`, `--subset`, `--ids-file`, `--prompt-variant`, `--option-perm`, `--backend text` (free text + fixed parser, `src/model/text.py`)
  - new per-score fields `tie` and `rendered_sha256`
  - a log-prob answer prefix (`logprob_prefill`, v2 templates only).
- **Analysis modules:**
  - `src/utils/scoring_extra.py`: threshold-free, soft-label, label-free decisions, risk–coverage, dev-fitted rules, PriDe
  - `src/utils/failure.py`: outcome and reason code per (model, post, label), token classes, lexicons, CMSB contrast set
  - additions to `stats.py` (simultaneous CIs, MDE, rank uncertainty, approximate randomization, Kendall's W, pigeonhole bootstrap)
  - additions to `evall.py` (hierarchical 1.3 decoding, `pyevall_scores`)
  - additions to `explain.py` (ERASER faithfulness, SHAP–LIME agreement, LIME stability, counterfactual rewrites, coding of self-explanations).
- **Notebooks:**
  - `99` post-hoc section: cells `ph-00`…`ph-18`, one per addendum section; the frozen cells are unchanged
  - new `30_encoder_baselines.ipynb` and `40_explainability.ipynb`
  - notebook 20 rebuilt around experiments, with batch presets in `docs/05`.
- **Paper skeleton:** `docs/report/sections_addendum.tex`, all numbers TODO, plus the Elsevier front and back matter items.
- **Packages:** PyEvALL 0.2.11 installed with `--no-deps`, plus `tabulate` and `jsbeautifier`. Its own pins (numpy 1.26 / pandas 2.2) conflict with ours. It reproduces the official dev majority-class baseline scores. Recorded in `requirements.txt` as comments.

**Checks.**
- `ruff` clean.
- **`pytest`: 58 tests** (new `tests/`). They cover:
  - experiment guards
  - unchanged frozen run names, v1 prompts and template sha256s against the stored predictions
  - the variant lock
  - every new metric against sklearn/scipy
  - failure codes, EvALL decoding, the PyEvALL baseline values
  - PriDe recovering a known prior
  - encoder training on a tiny local model
  - end-to-end `run_inference.py` sample runs with a stub model.
- **`99` with the new section:** all 44 non-PDF files of `results/final_eval/` regenerated byte for byte. The 9 PDFs embed creation dates and were not compared.
- **Full sanity chain on CPU with stub models:**
  - every experiment type as a sample experiment
  - notebook 30 (tiny RoBERTa) and notebook 40 as scripts
  - `99` in sanity mode twice
  - all cells ran and wrote 40 tables, with SANITY OK.
- The numbers from these checks were never read, and all verification outputs were deleted. The real `outputs/` and `results/` were not touched.

**Fixed along the way.**
- `compute_metrics` crashed on binary posts without hard gold. EXIST 1.1 dev has 45 such English tweets, so a dev run would have crashed after inference. It now skips them; CMSB/EDOS are unchanged.
- The hard-coded HF token in notebook 20 was replaced with `userdata.get`. It is in no commit, but must still be revoked.
- `CLAUDE.md`, `README.md` and `docs/04`–`11` now match the new scope.

**Deviations from the roadmap.**
- The roadmap's "refused → NO" EvALL variant was dropped: it fills a refusal with a class, against the integrity rule.
- E10 (counterbalanced few-shot sets) was not built. Few-shot is outside addendum A1, and the existing k1 sets must not be run.
- fp32 replication, contamination probes, HateCheck and guard models are not in A1.

**Next.**
1. The user commits `docs/12`, `configs/lexicons.yaml`, `configs/prompt_variants.lock`.
2. On Colab, each batch in sample mode then full, in this order: dev sweep, format check, answer prefix, option order (2 experiments), encoders (notebook 30), 5-prompt study.
3. `99` post-hoc run, which writes the explanation id lists.
4. Self-explanations (notebook 20, `ids_dir`), then notebook 40, then `99` again.
5. Fill `sections_addendum.tex` from `results/posthoc_eval/`.
