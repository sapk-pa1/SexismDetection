# 2 · Data pipeline: from raw files to processed Parquet

Notebook: `notebooks/01_data_prep.ipynb`. Config: `configs/data_prep.yaml`. Manifest: `results/data_manifest.json`.

The notebook reads the raw releases, builds one Parquet file per (dataset, split), checks every input and output against the committed manifest, and writes a run record. It is deterministic (nothing is random; the seed is set only so every notebook has the same cell layout).

## 2.1 Raw inputs

Datasets are not redistributed. They must sit under `data/` exactly like this (`00` checks the layout):

```
data/
  CMSB/sexism_data.csv                     13,631 posts, binary label
  CMSB/sexism_annotations.csv              28,250 rows, 5 annotators per annotated post
  EDOS/edos_labelled_aggregated.csv        20,000 posts, official split column
  EDOS/edos_labelled_individual_annotations.csv   60,000 rows, 3 annotators per post
  EXIST/EXIST 2025 Tweets Dataset/training/EXIST2025_training.json   6,920 tweets (en 3,260 / es 3,660)
  EXIST/EXIST 2025 Tweets Dataset/dev/EXIST2025_dev.json             1,038 tweets (en 489 / es 549)
  EXIST/EXIST 2025 Tweets Dataset/test/EXIST2025_test_clean.json     2,076 tweets (en 978 / es 1,098), no labels
  EXIST/evaluation/golds/EXIST2025_{training,dev}_task1_{1,2,3}_gold_{hard,soft}.json
  EXIST/evaluation/exist2025_format_val_V0.2.py   official submission validator (used in 99)
  splits/cmsb_split_s42.csv                frozen CMSB split (ids + split name only)
```

`01` hashes **20 raw files** (the 8 data files above plus 12 EXIST gold files: train/dev × tasks 1.1/1.2/1.3 × hard/soft). Their SHA-256 and byte sizes are in `results/data_manifest.json` under `raw`.

### Datasets and tasks

| Dataset | Source | Tasks used | Split |
|---|---|---|---|
| **CMSB** | "Call me sexist, but…" (Samory et al. 2021) | `binary` (`sexist`) | ours, frozen 70/10/20 |
| **EDOS** | SemEval-2023 Task 10 (Kirk et al. 2023) | `task_a` binary, `task_b` 4 classes, `task_c` 11 classes; B and C single-label, defined on sexist posts | official |
| **EXIST 2025** | CLEF EXIST 2025, tweets | `task_1_1` binary, `task_1_2` 4-class source intention (incl. `NO`), `task_1_3` multi-label over 5 categories | official; test unlabelled |

Each dataset is evaluated **on its own tasks**; there is no shared label space. Tasks, target columns and class order are declared once, under `tasks` in `configs/data_prep.yaml`:

```yaml
tasks:
  cmsb:  binary:   {type: binary, column: sexist}
  edos:  task_a:   {type: binary, column: sexist}
         task_b:   {type: multiclass, column: category, classes: [threats, derogation, animosity, prejudiced_discussions]}
         task_c:   {type: multiclass, column: vector, classes: ["1.1","1.2","2.1","2.2","2.3","3.1","3.2","3.3","3.4","4.1","4.2"]}
  exist: task_1_1: {type: binary, column: sexist}
         task_1_2: {type: multiclass, column: intention, classes: ["NO", REPORTED, JUDGEMENTAL, DIRECT]}
         task_1_3: {type: multilabel, column: labels, classes: [IDEOLOGICAL-INEQUALITY, STEREOTYPING-DOMINANCE, OBJECTIFICATION, SEXUAL-VIOLENCE, MISOGYNY-NON-SEXUAL-VIOLENCE]}
```
(Abbreviated; the real file is block YAML.) `"NO"` must be quoted: bare `NO` is YAML `false`. The EXIST 1.2 order follows the official validator. The EDOS Task C classes are the official vector codes; their names live in the prompt template and `data/CLAUDE.md`.

The same class order is repeated in `src/schemas/tasks.py` and `src/prompts/templates/{dataset}/{task}.yaml`. `check_against_config` and `PromptTemplate.load` fail on any drift ([04](04-llm-inference.md#three-place-class-order-invariant)).

## 2.2 Config (`configs/data_prep.yaml`)

| Key | Value | Meaning |
|---|---|---|
| `dataset, task, model, method, seed` | `all, prep, none, dataprep, 42` | Parts of the run name |
| `update_manifest` | bool | `false`: check mode (any hash mismatch stops the run). `true`: rebuild mode (prints what changed, rewrites the manifest) |
| `languages` | `[en]` | EXIST languages to keep. CMSB and EDOS are English |
| `cmsb_split_file` | `splits/cmsb_split_s42.csv` | Frozen split, never regenerated |
| `processed_dir` | `processed` | One subfolder per split |
| `manifest` | `results/data_manifest.json` | Relative to `PROJECT_ROOT`; committed |
| `parquet` | `compression: zstd`, `compression_level: 3` | Level set explicitly because codec defaults can change |
| `tasks` | see above | Task spec read by every later stage |

The notebook asserts the config has exactly these keys, that `update_manifest` is a bool, that `tasks` covers the three datasets and that every task has `classes` iff it is not binary.

> The committed config currently has `update_manifest: true`. It should be set back to `false` after an intended rebuild ([09](09-known-issues.md)).

## 2.3 Notebook walk-through

### Cells 1–5: paths, pins, config, seeds, environment
- Pins: pandas, numpy, scikit-learn, pyarrow, pyyaml (same as `00`). The installed versions must equal them.
- Run name: `{%Y%m%d-%H%M}_all_prep_none_dataprep_s42`. The time stamp prevents same-day reruns from overwriting a run; an existing run folder stops the notebook.
- `ENV` records Python and the pinned versions; `PKG_FREEZE` is a pip-freeze equivalent built from `importlib.metadata` (works without pip); `NOTEBOOK_SHA` is the SHA-256 of the notebook **as saved on disk** (save before running so it matches what ran); `config_sha256` hashes the YAML.

### Cell 7: raw-input check
1. Build the dict `RAW` of 20 keyed paths.
2. On Colab, copy them to `/content/data`.
3. SHA-256 every file and compare with `manifest["raw"]`.
   - No manifest and `update_manifest: false` → `FileNotFoundError` (usually a wrong `PROJECT_ROOT`).
   - Check mode → assert no raw file changed.
   - Rebuild mode → print which keys changed.
   - The manifest's recorded `env` is compared with the current one; a difference is a warning only.

### Cells 8–9: text normalisation
`text` is kept exactly as released. `text_norm` is the **one shared model input** for encoders and LLMs:

```python
URL_RE      = r"(?:https?://|www\.)\S+?(?=[)\]}>.,!?;:\"']*(?:\s|$))"   # stop before trailing punctuation
USER_RE     = r"(?<!\w)@\w+(?:@\w+)*|\bMENTION\d+\b"                    # @handles (incl. glued @a@b) and CMSB MENTION123
USER_RUN_RE = r"\[USER\](?:\s*\[USER\])+"                               # collapse runs of [USER]
GLUED_RT_RE = r"\bRT(?=MENTION\d+\b)"                                   # "RTMENTION12:" -> "RT MENTION12:"

normalise(text) = collapse_whitespace(
    USER_RUN_RE→[USER]( USER_RE→[USER]( URL_RE→[URL](
        GLUED_RT_RE→"RT "( NFC( html.unescape(text) ))))))
```

Why each step:
- The three datasets mark users and links differently (CMSB `MENTION123` and raw URLs; EDOS `[USER]`/`[URL]`; EXIST raw `@user`, raw URLs and HTML entities). All are mapped to EDOS's `[USER]`/`[URL]`.
- `html.unescape`: 287 English EXIST tweets had HTML entities.
- **NFC, not NFKC**, so stylised characters survive.
- The `(?<!\w)` lookbehind keeps Spanish inclusive forms like `tod@s`.
- `GLUED_RT_RE` fixes 2 CMSB posts where a retweet was glued to its mention.
- No lowercasing; emoji are kept.

Three asserts in the cell pin the behaviour on synthetic strings.

### Cells 10–11: schema and writing helpers
- `make_schema(extras)`: a fixed **pyarrow schema** per dataset: shared columns, then the dataset's own task columns ([2.4](#24-processed-files)).
- `cat_struct(cats, type)`: a struct with one field per class, in config order (soft shares and vote counts).
- `to_table`: sort by `(split, id)` (stable) for a deterministic row order; NaN becomes null under the explicit schema.
- `content_hash(table)`: SHA-256 of a JSON blob `{schema: [[name, type_sig, nullable], …], rows: table.to_pylist()}` with sorted keys. It hashes **content, not file bytes**, so it survives pyarrow or compression changes. `type_sig` avoids pyarrow's display strings. A pinned self-test hash on a fixed tiny table fails if the hashing itself ever changes.
- `write_parquet`: skips the write when the file on disk already has the same content hash; otherwise writes atomically (`.parquet.tmp` then `os.replace`), one row group, with a `heft` footer key holding `{pipeline, run_name, sources (raw SHA-256s), env, content_sha256}`. Because unchanged files are not rewritten, each file's footer describes the run that actually produced its content.

### Cells 12–13: CMSB
- Load `sexism_data.csv` (assert 13,631 unique ids) and `sexism_annotations.csv`.
- Drop the 4 **orphan** annotation ids not in the data file (3345, 4266, 9821, 11501); assert exactly these.
- Assert every annotated post has exactly 5 annotations.
- **Vote rule:** an annotator vote is sexist if `content ∈ {1,2,3,4}` **or** `phrasing == 1`. `sexist_soft` = mean of these votes. This rule reproduces the released `sexist` label on 100% of annotated rows at ≥ 3 of 5 (asserted in cell 15 as `sexist_soft >= 0.6`).
- Columns built: `id` (str), `text`, `text_norm`, `lang = "en"`, `sexist` (Int64), `sexist_soft`, `n_annot` (5 or null), `annotated` (has annotations), `of_id` (str id of the original for adversarial modifications; null for originals, which are `-1` in the raw file).
- Not kept: the source-subset column `dataset`, the content codes and Perspective `toxicity`. CMSB is binary only (decided 2026-10-19).
- The 7,985 `other` posts are **unannotated** (`annotated = False`, `sexist_soft`/`n_annot` null) and all non-sexist; asserted (`5,646` annotated posts).

### Cells 14–15: the frozen CMSB split
CMSB has no official split. `data/splits/cmsb_split_s42.csv` (columns `id, split`) is **the source of truth**:
- Built once with seed 42 using scikit-learn `StratifiedGroupKFold` (10 folds, assigned 7/1/2 → 70/10/20).
- **Grouped:** an original, its `of_id` modifications and exact-duplicate raw texts (case and whitespace ignored) share a split. 11,312 groups, the largest with 7 posts.
- **Stratified** on not-sexist versus the top-voted content category of sexist posts, so rare categories reach every split.
- **Cannot be regenerated:** the generating code was lost and a fresh `StratifiedGroupKFold` run does not reproduce the file. So `01` pins its SHA-256 (`7680236e…`, 147,023 bytes) and checks the properties instead. If raw data ever changes, make a new file with a new name.
- Read with `dtype=str, keep_default_na=False` so ids stay strings.

Checks in cell 15:
- Split ids equal the data ids, each once.
- Sizes are exactly train 9,543 / dev 1,363 / test 2,725.
- No `of_id` pair and no duplicate raw text crosses splits.
- The sexist rate differs by < 0.5 points across splits (all posts) and < 1 point (annotated only). Rates are **printed for train and dev only**: train/dev 0.1328 overall, 0.3204 / 0.3198 annotated-only (`outputs/20261019-1518_all_prep_none_dataprep_s42/metrics.json`).
- The vote rule reproduces the hard label; unannotated posts are all non-sexist.

**Main CMSB setting is annotated-only.** The `other` posts are 58.6% of train/dev, all non-sexist, from a different source, so including them rewards source detection. `run_inference.py` filters to `annotated == True` unless `--all-cmsb` is given (test: 1,126 of 2,725 posts).

### Cells 16–17: EDOS
- Load the aggregated file (hard labels) and the individual file (3 annotations per post). Assert unique ids, split sizes train 14,000 / dev 2,000 / test 4,000, and 3 annotations for every post.
- Map Task B names to config ids (`"1. threats, plans to harm and incitement"` → `threats`, `"2. derogation"` → `derogation`, `"3. animosity"` → `animosity`, `"4. prejudiced discussions"` → `prejudiced_discussions`); map Task C labels to their code (`"2.1 descriptive attacks"` → `2.1`).
- Consistency asserts, in **both** files: only known class names; `label_sexist == "sexist"` iff Task B is not `none`; each Task C vector sits under its Task B category (vector `2.x` under category `2`).
- Columns: `sexist` (Task A, Int64), `sexist_soft` (share of the 3 annotators voting sexist), `n_annot = 3`, `category` (Task B, null for non-sexist), `vector` (Task C code, null for non-sexist), `category_soft`/`category_votes` and `vector_soft`/`vector_votes` (structs over the classes: share and count out of 3).
- The aggregated hard label is **not** a plain majority vote: it disagrees with the 2-of-3 annotator majority on 772 of 16,000 train/dev posts (printed for train/dev only). The aggregated file is the hard target.
- Text already uses `[USER]`/`[URL]` placeholders.

### Cells 18–19: EXIST 2025
Targets come from the **official gold files**, never re-derived from raw annotations.
- For train/dev: load all six gold files (1.1/1.2/1.3 × hard/soft). Test has no golds.
- Asserts: soft golds cover every tweet; hard gold ids are a subset of the tweets; 1.2 hard values are in the config classes; **1.2 `NO` agrees with 1.1 `NO`** wherever both have a hard label; 1.1 hard values are only `YES`/`NO`; record ids and split tags (`TRAIN_EN`, …) are consistent; 1.3 hard is either `["NO"]` or has no `NO`.
- **Vote reconstruction, asserted equal to the soft gold exactly:**
  - 1.1: YES votes / `n_annot`, with `UNKNOWN` dropped.
  - 1.2: one label per annotator, `-` (not sexist) counted as `NO`, `UNKNOWN` dropped → `intention_votes`, `intention_n_annot`.
  - 1.3: each annotator gives a list; annotators with `UNKNOWN` dropped → `labels_votes`, `labels_n_annot`.
- Columns: `sexist` (1.1 hard, null without a majority), `sexist_soft` (1.1 soft YES), `n_annot` (1.1 annotator count), `intention` (1.2 hard, null without a majority), `intention_soft`, `intention_votes`, `intention_n_annot`, `labels` (1.3 hard list with `NO` dropped: `[]` = not sexist, null = no hard majority), `labels_soft`, `labels_votes`, `labels_n_annot`.
- Count asserts on the **full release before the language filter**: tweets per (split, lang) as in 2.1; 1.3 hard 6,050 train / 931 dev; 1.2 hard 5,496 / 841; 1.1 hard 6,064 / 934; test 0 for all.
- Then keep only `CONFIG["languages"]` (English): 3,260 / 489 / 978.
- `n_annot` and `labels_n_annot` differ on 53 English train tweets (1.3 `UNKNOWN` votes). Use `labels_n_annot`, not `n_annot`, for 1.3 vote counts, and don't compute 1.1 votes as `sexist_soft * n_annot` from 1.3 counts.

English 1.2 hard counts (`outputs/20261020-1100_all_prep_none_dataprep_s42/metrics.json`):

| split | NO | REPORTED | JUDGEMENTAL | DIRECT |
|---|---|---|---|---|
| train | 1,733 | 194 | 148 | 545 |
| dev | 250 | 35 | 28 | 87 |

### Cells 20–21: text audit flags
Computed on `text_norm` (the model input) for all three datasets. **Nothing is removed**; each experiment decides what to exclude and records it.
- `empty_norm`: the text is only `[USER]`/`[URL]` placeholders and whitespace.
- `dup_group`: a 12-hex-char id (prefix of SHA-256 of `text_norm`) shared by rows with identical `text_norm` within a dataset.
- `leak_group`: the same id, set only when that text appears in **more than one split**.

`AUDIT` counts flags per split and, on train/dev only, duplicate groups with conflicting `sexist`. Recorded values:
- CMSB: placeholder-only 67 / 12 / 16 (train/dev/test); leak rows 108 / 19 / 33; 8 duplicate groups with conflicting labels in train/dev. The leak exists only after normalisation (e.g. posts that differed only in the mention); raw-text grouping holds. Only **3** of the 33 test leak rows are among the 1,126 annotated posts that are scored.
- EXIST (en): 2 train and 1 test rows share a `text_norm` across splits (official split, flagged only).
- EDOS: no duplicates.

### Cells 22–23: build tables and check against the manifest
- Build the three schemas:
  - CMSB: shared + `annotated` (bool), `of_id` (str).
  - EDOS: shared + `category`, `vector` (str), `category_soft`/`vector_soft` (struct of float64), `category_votes`/`vector_votes` (struct of int64).
  - EXIST: shared + `intention` (str), `intention_soft`, `intention_votes`, `intention_n_annot`, `labels` (list<string>, child named `element` to match what Parquet reads back), `labels_soft`, `labels_votes`, `labels_n_annot`.
- **Config ↔ data check:** every task's column exists and holds only declared classes (binary: {0,1}; multilabel: exploded lists).
- Build 9 tables (3 datasets × 3 splits) with their source keys, compute `PROCESSED = {name: {rows, content_sha256}}`, and compare with `manifest["processed"]` **before anything is written**. In check mode any difference (or a different `languages`) stops the run, so a mismatch never overwrites good files.

### Cell 24: write
- Files go to `data/processed/{split}/{dataset}_{split}.parquet`. Files from the older flat layout are removed by exact name.
- Each written file is read back and its content hash asserted equal to the built one.
- The run folder (`exist_ok=False`) and the manifest are written together: `config.json` (config + env + notebook SHA), `pip_freeze.txt`, and in rebuild mode a new `results/data_manifest.json` (`written_by`, `env`, `languages`, `raw` {path, bytes, sha256}, `processed`). A copy of the manifest goes into the run folder as `data_manifest.json`.

### Cells 25–27: summary and run record
- `summarise` prints per-file counts. **Test files show only row counts and text flags**; train/dev also show sexist rates, annotated counts and per-task class counts. No test label statistic is ever printed.
- `metrics.json` holds `processed`, `written`, `AUDIT`, `STATS`; `README.md` states purpose, notebook and config hashes, mode and outcome.

## 2.4 Processed files

`data/processed/{train,dev,test}/{cmsb,edos,exist}_{split}.parquet`: one row per post, sorted by `(split, id)`, zstd level 3.

| File | Rows |
|---|---|
| `cmsb_train` / `cmsb_dev` / `cmsb_test` | 9,543 / 1,363 / 2,725 (annotated: 1,126 in test) |
| `edos_train` / `edos_dev` / `edos_test` | 14,000 / 2,000 / 4,000 |
| `exist_train` / `exist_dev` / `exist_test` | 3,260 / 489 / 978 (English only; test labels all null) |

### Shared columns

| Column | Type | Meaning |
|---|---|---|
| `id` | str | CMSB `id`, EDOS `rewire_id`, EXIST `id_EXIST` |
| `text` | str | unchanged post text |
| `text_norm` | str | normalised model input |
| `lang` | str | `en` / `es` |
| `split` | str | `train` / `dev` / `test` |
| `sexist` | int64, nullable | hard binary label (CMSB released, EDOS Task A, EXIST 1.1 hard) |
| `sexist_soft` | float64, nullable | share of annotators judging the post sexist |
| `n_annot` | int64, nullable | annotators behind `sexist_soft`; null for CMSB `other` and EXIST test |
| `empty_norm` | bool | placeholder-only input |
| `dup_group` | str, nullable | duplicate `text_norm` id within the dataset |
| `leak_group` | str, nullable | same id, only when the text crosses splits |

### Per-dataset columns

| Dataset | Column | Type | Meaning |
|---|---|---|---|
| CMSB | `annotated` | bool | False for the 7,985 `other` posts |
| CMSB | `of_id` | str, nullable | original post this row modifies |
| EDOS | `category` | str, nullable | Task B class; null for non-sexist |
| EDOS | `vector` | str, nullable | Task C code `1.1`–`4.2`; null for non-sexist |
| EDOS | `category_soft` / `category_votes` | struct | per Task B class: share / count of 3 |
| EDOS | `vector_soft` / `vector_votes` | struct | per Task C class: share / count of 3 |
| EXIST | `intention` | str, nullable | 1.2 hard label; null without a majority |
| EXIST | `intention_soft` / `intention_votes` | struct | 1.2 soft gold / per-class count |
| EXIST | `intention_n_annot` | int64, nullable | 1.2 annotators behind the votes |
| EXIST | `labels` | list<string>, nullable | 1.3 hard labels; `[]` = not sexist; null = no hard majority |
| EXIST | `labels_soft` / `labels_votes` | struct | 1.3 soft gold / per-category count |
| EXIST | `labels_n_annot` | int64, nullable | 1.3 annotators behind the votes (4–6 in train) |

Reading: `pd.read_parquet(path)` returns `labels` as numpy arrays, structs as dicts, and `sexist`/`n_annot` as pandas `Int64`.

### Soft labels

"Soft" is **not** the same quantity across datasets. Normalise explicitly from votes and counts before using one soft loss or metric across datasets.

| | `sexist_soft` | Category shares | Hard vs soft |
|---|---|---|---|
| CMSB | share of 5 voting content 1–4 or phrasing 1 | none | `sexist` = `sexist_soft ≥ 0.6`, asserted |
| EDOS | share of 3 voting sexist | one class per annotator, so a row sums to at most `sexist_soft` | hard label is not a plain majority (772 / 16,000 train/dev disagree) |
| EXIST | 1.1 soft gold YES (6 annotators) | 1.3: multi-select, rows can sum to > 1 | official golds; votes / count equals the soft gold exactly |

## 2.5 Manifest and hashing

`results/data_manifest.json` (committed):
```json
{ "written_by": "20261020-1100_all_prep_none_dataprep_s42",
  "env": {"python": "3.12.15", "pandas": "3.0.6", "numpy": "2.5.4", "pyarrow": "26.0.0", "pyyaml": "6.0.3", "scikit-learn": "1.9.1"},
  "languages": ["en"],
  "raw": {"cmsb_data": {"path": "...", "bytes": ..., "sha256": "..."}, ... 20 entries},
  "processed": {"cmsb_train": {"rows": 9543, "content_sha256": "..."}, ... 9 entries} }
```

Rules:
- Every run of `01` checks raw hashes, then processed content hashes, before writing.
- Mismatch in check mode stops the run. Set `update_manifest: true` **only for an intended change**; the run prints what changed. Set it back to `false` afterwards and run once more in check mode; log both runs.
- Any change to `text_norm`, columns or types changes the processed hashes and needs a rebuild.

## 2.6 Build history

| Run | Mode | What changed |
|---|---|---|
| `20261019_all_prep_none_dataprep_s42` | — | Old schema; kept as history |
| `20261019-1518_…` | rebuild | Per-dataset schemas, audit flags; all 9 processed hashes changed |
| `20261019-1519_…` | check | Everything matched |
| `20261019-1557_…` | rebuild | Rebuilt `processed/` (9 of 9 files written) after it was deleted; its README says it rewrote the manifest. Not in the experiment log |
| `20261020-1100_…` | rebuild | EXIST 1.2 columns, `n_annot` vs `labels_n_annot`, glued-RT fix; only `exist_*` changed; 4 new raw entries (1.2 golds). **Current manifest.** |

Between 2026-10-19 15:19 and 15:38 the CMSB split file and `data/processed/` were deleted; a byte-identical split copy (hash matches) was recovered from a scratchpad and `processed/` was rebuilt. Details in the experiment log.

## 2.7 Test-set handling

- Test files are built by the same code as train/dev, so inputs and schema match exactly, and they are covered by the manifest.
- `01` prints no test label statistic.
- **Test labels are read only in `99_final_eval`.** Label-blind reads of test text are allowed in `02_data_analysis` (`TEST_COLS`) and `run_inference.py --split test` (only `id, text_norm, split`, plus CMSB `annotated`).
- EXIST test has no public gold; it is scored by EvALL. Test ids: Spanish 500001–501098, English 600001–600978.

## 2.8 Open data decisions
- Which flagged CMSB rows to exclude (placeholder-only, conflicting duplicates, `of_id` modifications): undecided; flagged only.
- EDOS B/C evaluation scope: resolved for the LLM study as **gold-sexist posts** (SemEval setting); still marked TODO in `configs/data_prep.yaml`.
- Spanish EXIST: add `es` to `languages` and rebuild with `update_manifest: true`.
