# 3 · Data analysis (`02_data_analysis.ipynb`)

Descriptive statistics of the processed files written by `01`. Runs on CPU. No post text is printed.

- **Inputs:** `data/processed/{train,dev,test}/{cmsb,edos,exist}_{split}.parquet`; the `tasks` block of `configs/data_prep.yaml`.
- **Outputs:** `OUTPUT_ROOT/{run_name}/` with CSV tables, PNG figures in `figures/`, `analysis.md`, `findings.md`, `metrics.json`, `config.json`, `env.json`, `pip_freeze.txt`, `README.md`.
- **Run name:** `{%Y%m%d-%H%M}_all_analysis_none_eda_s42`. An existing folder stops the run.
- **Pins:** pandas, numpy, pyarrow, pyyaml (as in `00`) plus matplotlib 3.11.2.

## Test is read label-blind

Test files are opened with `TEST_COLS` only:

| Dataset | Test columns read |
|---|---|
| CMSB | `id, text_norm, lang, empty_norm, dup_group, leak_group, of_id` |
| EDOS | `id, text_norm, lang, empty_norm, dup_group, leak_group` |
| EXIST | same as EDOS |

An assert checks that none of `sexist, sexist_soft, n_annot, annotated, category, vector, labels` is in `TEST_COLS`. CMSB `annotated` and `n_annot` are left out because they mark the all-non-sexist `other` source and so give away the label. Every label statistic is train/dev only. (This is a recorded deviation from "test is read only in 99", requested by the user; the rule now allows label-blind reads.)

Each file's `content_sha256` from its `heft` footer is stored in `env.json`, so the statistics can be traced to an exact build.

## Sections

| § | Content | Tables / figures |
|---|---|---|
| 1 | Rows per split and language; share of the dataset | `size.csv` |
| 2 | Text on all splits: word counts (mean, median, p95, max) on `text_norm`, median chars, share with `[USER]`/`[URL]`, audit-flag counts | `text.csv`, `figures/length_by_split.png` (share of posts per 2-word bin, clipped at 70 words) |
| 3 | Binary label (train/dev): hard rows, sexist rate, unanimity share (soft 0 or 1), mean soft; CMSB also on annotated only. Crosstab of vote share vs hard label | `binary.csv`, `binary_soft_vs_hard.csv`, `figures/sexist_soft_by_split.png` |
| 4 | CMSB structure: originals vs adversarial modifications per split; sexist rates (train/dev) | `cmsb_structure.csv` |
| 5 | EDOS Task B/C class counts and shares over sexist posts; "category agreement" = share of sexist posts whose hard Task B class got ≥ 2 (and = 3) of 3 votes | `edos_task_b.csv`, `edos_task_c.csv`, `edos_agreement.csv`, `figures/edos_b_c_by_split.png` |
| 6 | EXIST 1.3 (train/dev): no-majority count, hard not-sexist / sexist, 1.1 vs 1.3 mismatches, `n_annot ≠ labels_n_annot`; label frequency, cardinality, mean soft share; co-occurrence P(column | row) on hard-labelled sexist train tweets | `exist_summary.csv`, `exist_label_freq.csv`, `exist_cardinality.csv`, `exist_soft_mean.csv`, `exist_cooccurrence_train.csv`, `figures/exist_cooccurrence_train.png` |
| 7 | Length by label (train; CMSB annotated only): median and mean words for sexist vs not | `length_by_label_train.csv` |

Report files:
- `analysis.md`: every table as Markdown (a small `md_table` helper; no `tabulate` dependency) with links to the figures.
- `findings.md`: fixed wording with **every number computed from this run's tables**, so the text cannot drift from the data.

## Runs

| Run | Note |
|---|---|
| `20261019-1618_all_analysis_none_eda_s42` | First run (a scratch run before it gave an identical `metrics.json`) |
| `20261019-1619_all_analysis_none_eda_s42` | Folder present; not in the experiment log |
| `20261019-1623_all_analysis_none_eda_s42` | Adds `analysis.md` and `findings.md`; same numbers as 1618. **Reference run.** |

All three runs predate the EXIST 1.2 columns (added 2026-10-20), so `02` has no 1.2 section yet.

## Key findings (`outputs/20261019-1623_all_analysis_none_eda_s42/findings.md`)

Observations, train / dev unless stated:
- **Sexist rate:** CMSB 13.3% / 13.3% (32.0% / 32.0% annotated only); EDOS 24.3% / 24.3%; EXIST 39.6% / 43.7% over tweets with a hard 1.1 label (390 / 45 have no majority).
- **Unanimous annotators:** CMSB 66.0% / 62.2%, EDOS 77.5% / 79.5%, EXIST 33.4% / 34.6%.
- **EDOS hard vs 2-of-3 majority:** differ on 697 / 14,000 train and 75 / 2,000 dev posts. Every vote share maps to one hard label for CMSB and EXIST, not for EDOS.
- **Text (all splits):** median words CMSB 14, EDOS 21–22, EXIST 25–27; longest post 63 words. `[USER]` in 38–39% of CMSB, 5% of EDOS and 55–59% of EXIST posts.
- **Audit flags** (train/dev/test): CMSB duplicate rows 543 / 76 / 158, leak rows 108 / 19 / 33; EXIST duplicates 4 / 0 / 1, leak 2 / 0 / 1; EDOS none.
- **CMSB:** adversarial modifications 1,601 / 230 / 461; train sexist rate 15.5% on originals vs 2.1% on modifications.
- **EDOS Task B** (train shares of sexist posts): derogation 46.8%, animosity 34.3%, prejudiced discussions 9.8%, threats 9.1%. The hard class has ≥ 2 of 3 votes on only 70.8% / 72.2% of sexist posts.
- **EDOS Task C dev** has fewer than 10 examples for 1.1 (8), 3.3 (9) and 3.4 (7).
- **EXIST 1.3 train:** 1 label 32.3%, 2 labels 46.2%, 3 labels 16.5%; P(STEREOTYPING-DOMINANCE | IDEOLOGICAL-INEQUALITY) = 0.63, P(OBJECTIFICATION | SEXUAL-VIOLENCE) = 0.61.
- **Length by label** (median words, not sexist vs sexist): CMSB 16 vs 15, EDOS 22 vs 24, EXIST 26 vs 26.

Hedged interpretation from the run: EXIST has the lowest agreement, so soft labels may carry most information there; length gaps are small, so length alone looks unlikely to be a strong shortcut (only a length-only baseline would show it); dev thresholds for the small EDOS C classes would rest on very few examples.

## Ad-hoc checks (2026-10-19, not a pipeline run)
Logged in the experiment log: no leftover URLs or `@handles` after normalisation; CMSB format cues (`RT` prefix in 8.4% of annotated vs 0.3% of `other` posts; hashtags 21.0% vs 43.7%) are extra source cues in the all-rows setting; length p99/max in words: CMSB 28/41, EDOS 48/58, EXIST-en 55/63.
