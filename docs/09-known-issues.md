# 9 · Known issues, open decisions and TODOs

Status as of 2026-10-23, from reading the code, configs and run folders. Items marked **(new)** were found while writing these docs. Items marked **(fixed 2026-10-23)** were resolved by the addendum A1 implementation ([12](12-protocol-addendum.md), experiment log 2026-10-23).

## Needs attention now

- **(fixed 2026-10-23) HF token hard-coded in `notebooks/20_llm_runs.ipynb`.** The cell is back to `userdata.get("HF_TOKEN")`. `git log -S` finds the token in no commit. **Still to do:** revoke it on the Hub, because it sat in a Drive notebook and appeared in a session transcript, and check Drive copies.
- **(new) `configs/data_prep.yaml` has `update_manifest: true`.** The rule is to set it back to `false` after an intended rebuild, so the next run checks instead of rewriting.
- **(new) Stray prediction files at the repo root.** `predictions.jsonl` is tracked in git (a Gemma CMSB run) and `predictions (1).jsonl` is untracked. Run outputs belong in `OUTPUT_ROOT/{run_name}/`, not in git.

## Protocol deviations (to state in the paper)
- No dev runs for the primary numbers: thresholds (P(Yes)/mass ≥ 0.5, argmax) and the soft-refusal cutoff (0.5) were fixed a priori. Dev runs are allowed since 2026-10-23 for pre-registered secondary analyses only (docs/12 §3.1).
- One seed and one prompt (version 1) per task in the primary numbers; the five-prompt study is pre-registered (docs/12 §3.6), not yet run.
- **(2026-10-23)** The CMSB test composition (`of_id`) was inspected after primary scoring, during the research sweep (experiment log 2026-10-23). The CMSB contrast-set analysis is post-hoc.
- Qwen3.5-4B thinking-on not run: it needs a generate-then-score path, because JSON-constrained decoding from the first token suppresses thinking.
- EXIST: English only; test scores pending EvALL; the official validator expects 2,076 (es + en) items.
- EvALL files were written to the run folder, not `results/evall_submissions/` as the protocol entry said (size, ~13 MB).
- Option-order position bias is not handled in the primary lettered-option numbers (e.g. on EDOS B 9 of 15 models never predict the last option, prejudiced discussions). Option-order control is pre-registered (docs/12 §3.5).
- **(2026-10-23)** Exact ties between option letters go to the earlier-shown class (534 of 60,000 EDOS C test predictions; found by the research sweep). New runs record `tie` per score.

## Open decisions
- Which flagged CMSB rows to exclude (placeholder-only, conflicting duplicates, `of_id` modifications) for the supervised models.
- EDOS B/C evaluation scope for the supervised models (the LLM study used gold-sexist posts; `configs/data_prep.yaml` and `load_split` still carry TODOs).
- EDOS Task C thresholding given dev classes with < 10 examples.
- **(fixed 2026-10-23)** Local ICM / ICM-soft: `pyevall_scores` (PyEvALL 0.2.11, installed with `--no-deps`) scores EXIST dev. The hard metrics use the hard-majority subset (n = 444/400/444), the soft metrics all 489 English dev tweets.
- EvALL: the team name is a TODO in both `CONFIG["evall_team"]` (frozen) and `PH["evall_team"]`. Whether English-only runs are accepted is unknown (English-only predictions against the bilingual gold score ICM-Norm 0.47 even when perfect), so the post-hoc export stays off until the organisers confirm.
- **(decided 2026-10-23, D1)** HEFT/NeoBERT move to a follow-up paper; this paper is the standalone zero-shot study with an encoder reference.
- Whether to commit `data/splits/cmsb_split_s42.csv` (ids and split only; it can't be regenerated) and how others can obtain it.
- Spanish EXIST.

## Known technical issues
- **(fixed 2026-10-23)** `compute_metrics` crashed on binary posts without hard gold (EXIST 1.1 dev has 45 English tweets with no majority). They are now skipped like in the other branches; CMSB/EDOS results are unchanged.
- **(2026-10-23)** The EXIST 1.3 soft "NO" in the frozen EvALL export is Π(1 − P(Yes)), but the official gold has 1.3 NO = 1.1 NO. The hierarchical decoding (docs/12 §2.9) fixes it for new exports; the frozen files keep the old rule.
- **bf16 score grid.** Logits are computed in bf16 and only cast to float32 before the softmax, so P(Yes) takes few distinct values, giving a coarse score grid and possible exact 0.5 ties (seen in the first CMSB dev Gemma run; one EXIST 1.3 test score of Gemma is exactly 0.5). Float32 last-position scoring in `src/model/logprob.py` is pending. `99` reports `n_distinct_scores` per model.
- **No `mamba-ssm`/`causal-conv1d`.** Jamba2 and Falcon-H1 use the PyTorch Mamba path: slower, and numbers can differ slightly from the CUDA kernels.
- **Ministral 3 FP8** is dequantized to bf16 at load; tested only through the actual runs.
- **Generate backend:** can't show refusals, can't restrict JSON whitespace, and on a random model filled the token budget with whitespace. Not used for any reported number.
- `environment.json` is rewritten on each resume of a run.
- The full prompt text is not stored per prediction; it is rebuilt from the template SHA-256 and the example id.
- Several EDOS definitions in the templates were excerpted in our copy of the paper and are marked TODO.

## Stale or inconsistent text (new)
- `README.md` lists `99_final_eval` as "(planned)"; it exists and has run. Its "Results: TODO" can now point to `results/final_eval/`.
- `data/README.md` "Datasets at a glance" lists EXIST tasks 1.1 and 1.3 only, and its task-column table lacks the 1.2 `intention*` columns and `labels_n_annot`.
- `01_data_prep` intro cell lists EXIST 1.1 and 1.3 only; its Findings cell still reports the split file as missing and the rebuild as pending (both resolved).
- `00_colab_setup` says Gemma is gated (it isn't; Llama and Aya are). Its file check doesn't include the EXIST test file or the 1.1/1.2 golds.
- `20_llm_runs` Summary cell is still TODO; it ends with ad-hoc diagnostic cells and two empty cells; `dry_run` is saved as `False` although the docs say `True` is the default.
- `src/utils/metrics.py` `_decision_rule` and `logprob.py` still describe 0.5 as "provisional until tuned on dev"; under the test-only decision it is the final, a-priori rule.
- `ChatOptions.prefill` comment mentions Granite Guardian, which was removed.
- `pyproject.toml` references a `requirements.lock` that doesn't exist.
- The Colab runtime for the test batch reported Python 3.13.15, while `pyproject.toml` requires `>=3.12,<3.13`. The results don't depend on it, but `environment.json` should be cited when describing the setup.
- scipy, matplotlib and jsonschema are used by `99` but not pinned.
- `01` (~8 helpers) and `02` exceed the 3-helper limit of `.claude/rules/code-structure.md`; moving them into `src/` needs a before/after hash check.
- Run folders `20261019-1557_all_prep_none_dataprep_s42` and `20261019-1619_all_analysis_none_eda_s42` exist but have no experiment-log entry.

## Not started
- Encoders (RoBERTa, DeBERTa-v3, HateBERT), vanilla NeoBERT and HEFT (`1x` notebooks). HEFT's description is TODO everywhere.
- Explainability runs (`3x`), with a Captum IG cross-check once encoders exist.
- Statistics notebooks (`4x`) beyond what `99` already does.
- **(fixed 2026-10-23)** Tests: `tests/` has 58 tests (`pytest`). They cover experiment folders, run names, prompt variants and the lock, every new metric (checked against sklearn/scipy), failure codes, EvALL decoding and PyEvALL, and end-to-end sample runs of `run_inference.py` with a stub model.
