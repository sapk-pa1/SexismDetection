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
