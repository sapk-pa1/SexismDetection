# 12 · Protocol addendum A1 (2026-10-23): post-hoc analyses, new runs, failure analysis, explainability

This addendum fixes every rule for the analyses added after the frozen test evaluation of 2026-10-20 ([06](06-final-evaluation.md)). It is written and committed **before** any of these analyses or runs. Rules here may not be changed after their results are seen; a change needs a new dated addendum and new run names.

**Status labels used in the paper and the tables.**
- **Primary:** the frozen 2026-10-20 protocol and numbers. Unchanged.
- **PH (post-hoc secondary):** new analyses of the existing test predictions. Test labels were read on 2026-10-20, so these are explicitly post-hoc.
- **PR (pre-registered secondary):** new inference runs whose rules are fixed here before they run.
- Every table and figure from this addendum carries its label. The paper puts them in a section titled "Analyses added after test labels were read (2026-10-20)".

**Decisions behind this addendum** (user, 2026-10-23):
- D1: a standalone zero-shot paper; HEFT moves to a follow-up.
- D2: label-blind dev inference is allowed, as PR. This reverses the 2026-10-20 test-only rule for secondary analyses only; the primary numbers stay test-only.
- EXIST stays, scored on its labelled English dev split.
- GPU scope: format check, answer prefix, option order, a 5-prompt study and encoder baselines.

**Where things run.**
- Test labels are read only in `notebooks/99_final_eval.ipynb`, in its new section "Post-hoc analyses (addendum A1)". The frozen sections 1–12 of `99` are not edited.
- New inference runs go to experiment folders `OUTPUT_ROOT/experiments/{YYYYMMDD-HHMMSS}_experiment_{name}/` (see `src/utils/experiments.py`). `99` reads them only through each folder's `manifest.csv`.
- The frozen 105-run batch is registered read-only as experiment `20261020-000000_experiment_zeroshot-frozen`, without moving or changing any file.
- Sample (sanity) runs go to separate `{name}-sample` experiments, which `99` refuses.

## 1. Shared definitions
- **Log-odds** of a Yes/No label: s = ln P(Yes) − ln P(No), from the stored next-token probabilities. Each probability is clipped to [1e−12, 1] first, so s is finite.
- **Primary decision:** Yes iff P(Yes)/mass ≥ 0.5, i.e. s ≥ 0, unchanged. The **primary non-answer** rule is unchanged too: mass < τ = 0.5.
- **Answer-forced distribution:** answer-token probabilities renormalised over the answer set, ignoring mass (the τ = none variant of [06](06-final-evaluation.md)).
- **Bootstrap:** percentile intervals, B = 10,000, seed 42, resampling posts. The resample matrix is shared by all models of a task, as in the frozen protocol. Exceptions are stated per item.
- **Train priors** (computed 2026-10-23 from the train parquet):
  - CMSB annotated 0.3204; EDOS A 0.2427; EXIST 1.1 0.3962
  - EDOS B: derogation 0.4679, animosity 0.3428, prejudiced discussions 0.0980, threats 0.0912
  - EDOS C and EXIST 1.2 shares come from the same files, computed in code.
- **Lexicons:** `configs/lexicons.yaml` v1, frozen with this addendum: gender terms, gendered slurs, profanity, the gender-swap map and the non-answer token classes. Matching is lower-cased, whole-word.

## 2. Post-hoc analyses of the existing test predictions (PH)

### 2.1 Statistics display
- **Simultaneous Δ-vs-best intervals:** the 0.18th and 99.82th percentiles of the shared bootstrap Δ (Bonferroni, α = .05 over 14 comparisons). They are shown next to the per-comparison 95% interval. The frozen Holm decision ("not separable") is unchanged and gets a footnote saying the dagger is family-wise.
- **p-values:** p_boot = 0 is printed as "< 1e-4" (B = 10,000).
- **MDE** (minimum detectable effect) per comparison: (z₀.₉₇₅ + z₀.₈) × SD of the bootstrap Δ. Also reported at the Holm first step: (z_{1−.05/28} + z₀.₈) × SD. The median per task goes in the tables. Wording for ties: "not separable at this test size (MDE ≈ x)".
- **Rank uncertainty:** P(best) is the share of resamples in which the model has the highest macro-F1, with ties split equally. The 95% rank interval comes from the 2.5–97.5th percentiles of the model's rank across resamples (rank 1 = best; ties take the average rank).
- **Size vs F1:** Spearman ρ with a Fisher-z 95% CI, SE = √(1.06/(n−3)), n = 15. Holm over the 4 labelled test tasks. Descriptive only.
- **Cross-task ranking agreement:** Kendall τ_b between models' macro-F1 rankings for every pair of labelled tasks, with a bootstrap CI (B = 2,000, posts resampled within each task). Descriptive.
- **Approximate randomization:** a paired check beside p_boot. R = 10,000, seed 42, the two models' predictions swapped per post with p = .5, p = (1 + #{|d*| ≥ |d|})/(R + 1), Holm over 14. We report whether any frozen "not separable" mark would change; the frozen marks are not altered.

### 2.2 Threshold-free and proper-score metrics
- **Binary tasks (CMSB, EDOS A):**
  - AUROC of s, with ties counted as 0.5 (sklearn `roc_auc_score`)
  - AUPRC, i.e. average precision of s (sklearn `average_precision_score`)
  - Brier of P(Yes)/mass against the hard gold
  - MCC under the primary decision (strict: a non-answer is the wrong class).

  Each with bootstrap CIs and the paired Δ vs the best model by that metric (Holm over 14).
- **EDOS B/C:** per-class F1 with gold support n and a 95% bootstrap CI.

### 2.3 Soft-label (annotator-disagreement) metrics
- **Gold:**
  - CMSB `sexist_soft`: the share of 5 annotators meeting the Samory et al. rule.
  - EDOS `sexist_soft`: the share of 3 annotators.
  - EDOS B/C: `category_soft` / `vector_soft` renormalised to sum to 1 over sexist votes. Gold-sexist posts with zero sexist votes (expert overrides) are excluded from the B/C soft metrics and counted.
- **Prediction:** the answer-forced distribution for every scored post, with the strict non-answer rate reported beside it. No value is filled in for non-answers.
- **Metrics:**
  - **primary soft metrics:** mean L1 (binary |q − s|; multiclass Σ_c |q_c − s_c|) and mean Brier (binary (q − s)²; multiclass Σ_c (q_c − s_c)²)
  - **appendix only:** cross-entropy −Σ s_c ln max(q_c, 1e−6). Cross-entropy is not primary: it is dominated by the clipping ε on saturated scores (LeWiDi-2025).
- **Reference rows:**
  - constant train prior
  - hard gold used as the prediction
  - the 15-model mean distribution (a "crowd" of models).
- **Hard-vs-soft ranking agreement:** Kendall τ_b between the macro-F1 and L1 rankings.

### 2.4 Label-free decision rules (binary tasks)
- **Prior-matched rule:** label the top round(π_train · n) posts by s as Yes. Ties are broken by post id order, so the predicted rate equals the train prior. The rule is re-applied inside every bootstrap resample. It is reported as a measure of ranking quality (the precision–recall break-even point). Train prevalence is close to test prevalence on these datasets (CMSB 0.3204 vs 0.3206; EDOS A 0.2427 vs 0.2425 = 970/4,000), which the text must say.
- **Plain batch calibration:** predict argmax_c [ln q_c − ln q̄_c], where q̄ is the mean answer-forced distribution over the task's evaluated posts.
- **Prior-aware batch calibration:** the same plus ln π_c (train prior).
- Both are also applied to the EDOS B/C letters, with train class shares among gold-sexist train posts. Both are re-estimated inside each resample. Report macro-F1 with CIs and the predicted positive rate.

### 2.5 Abstention views
- **Risk–coverage:**
  - Confidence = |s| for binary tasks, and the maximum answer-forced class probability for option tasks (plus a mass-based curve for option tasks).
  - Selective risk = 1 − accuracy on the covered posts. AURC = the mean selective risk over coverages k/n, k = 1..n. Also risk at 80% coverage.
- **Pairwise fair subsets:** for each model vs the best, macro-F1 of both on the posts both answered at τ = 0.5, with a paired bootstrap Δ CI.

### 2.6 Failure analysis: "why the model did not say Yes"
For every (model, scored post, label) under the primary decision, `src/utils/failure.py` records an **outcome** and **reason codes**.
- **Outcome:** TP / FP / FN / TN for binary and multilabel; correct / wrong / non-answer for multiclass.
- **Reason codes, binary and multilabel:**

  | Code | Rule |
  |---|---|
  | `non_answer` | mass < 0.5 |
  | `confident_no` | s ≤ −2 |
  | `borderline_no` | −2 < s < 0 |
  | `confident_yes` | s ≥ 2 |
  | `borderline_yes` | 0 ≤ s < 2 |
  | `exact_tie` | P(Yes)/mass exactly 0.5 |

- **Reason codes, multiclass:**

  | Code | Rule |
  |---|---|
  | `non_answer` | mass < 0.5 |
  | `confident_wrong` | the predicted class holds ≥ 0.8 of the answer-forced mass |
  | `near_tie` | top-2 gap < 0.05 |
  | `tie_break` | the top two are exactly equal; the earlier class wins |
  | `wrong` | otherwise |

- **Non-answer token class:** the stored top-1 token, classed with the `non_answer_tokens` lists in `configs/lexicons.yaml`. Classes in order: answer token, markup (no letter or digit), refusal, none-of-options, explanation, other.
- **Context per post**, which explains the misses:
  - annotator support (soft share) and `n_annot`
  - EDOS expert override: gold label disagrees with the annotator majority
  - gold fine-grained class: EDOS B category and C vector; EXIST 1.2 intention and 1.3 categories (on dev)
  - CMSB source (`dataset` column of the raw CSV: callme / hostile / benevolent / scales) and original vs adversarial modification (`of_id`)
  - lexicon hits: gender term, gendered slur, profanity
  - length in words: ≤ 10 / 11–25 / > 25.
- **Aggregates**, each with 95% bootstrap CIs over posts; descriptive, with no new significance claims:
  - **False-negative rate (FNR)** by sexism form:
    - EDOS B category and C vector
    - CMSB source, e.g. benevolent vs hostile items
    - explicit vs implicit, using EDOS derogation + threats vs animosity + prejudiced discussions (Kirk et al.: derogation is explicit, animosity implicit), and slur present vs absent
    - EXIST dev intention (DIRECT vs REPORTED vs JUDGEMENTAL)
  - FNR and **false-positive rate (FPR)** by annotator-support bin: 0, (0, 0.5), [0.5, 1), 1. This answers "whose sexism do models flag or miss?".
  - Reason-code shares among misses, per model.
  - FPR on non-sexist posts with vs without gender terms.
  - Predicted vs gold positive rate per dataset: EXIST under-prediction vs CMSB/EDOS over-prediction.
- **Outputs:**
  - per model, a `failure_reasons.parquet` file in the post-hoc run folder (ids, codes and context only, no text)
  - aggregates in `results/posthoc_eval/tables/failure_*.csv`.

### 2.7 Label-free reference baselines (scored like a model)
- **Gender-term rule:** sexist iff the post contains any gender term.
- **Perspective toxicity** ≥ 0.5 (CMSB only, from the raw CSV's `toxicity` column), plus AUROC of the toxicity score.

### 2.8 CMSB contrast set
- **Pairs:** an original o and each adversarial modification m with `of_id` = o, both among the 1,126 scored posts.
- **Metrics:**
  - true-negative rate on modifications vs on other non-sexist posts, and recall on originals
  - **pair accuracy:** o predicted sexist and m not, over pairs whose gold is o sexist, m not sexist (other pairs excluded and counted)
  - **within-pair log-odds gap** s(o) − s(m): mean, with a pair-level bootstrap CI
  - the number of pairs, also per source.
- **Note for the paper:** the test composition was inspected on 2026-10-23 after primary scoring (see the deviation in the experiment log).

### 2.9 EXIST test export to EvALL (only if the organisers confirm English-only scoring)
- **Variant names:** `omit` (formerly `strict`): non-answers left out, which EvALL scores as "no category" (lenient). `answer-forced` (formerly `nocut`): τ = none.
- **No refused → `NO` variant.** Filling a refused label with a class would break the refusal rule in `.claude/rules/research-integrity.md`. This deviates from roadmap item E2. Our own strict scoring stays macro-F1 with a non-answer counted as wrong.
- **1.3 hierarchical decoding:**
  - Hard: `["NO"]` if the model's own 1.1 decision is No. Otherwise the categories with P(Yes)/mass ≥ 0.5; if none pass, the single highest-scoring category.
  - Soft: `NO` = the model's 1.1 P(No)/mass; each category = its own P(Yes)/mass. This replaces Π(1 − P).
- 1.2 stays independent. The share of tweets where 1.2 = NO but 1.1 = Yes (or the reverse) is reported label-free.

## 3. New inference runs (PR)
Every run in this section uses:
- all 15 models in `configs/llms.yaml`, at their pinned revisions
- the log-prob backend unless stated
- seed 42
- the frozen v1 templates unless stated.

Each batch first runs in sample mode (`CONFIG["sample"] = 5`, SANITY OK required), then in full.

### 3.1 Dev sweep (`experiment: devsweep`)
- **Tasks and posts:** all 7 tasks on dev, with v1 prompts. CMSB annotated (566 posts); EDOS A (2,000); EDOS B/C on gold-sexist dev posts (486); EXIST English dev (489 tweets).
- **Uses**, all reported as PR rows next to the primary numbers:
  - **(a) Dev-tuned log-odds threshold** per (model, binary or multilabel label). Candidates are the midpoints between sorted unique dev values of s plus 0. Pick the one that maximises dev macro-F1 for that label; ties go to the smallest |t|. Apply it to test.
  - **(b) Dev-chosen non-answer cutoff τ** ∈ {0.1, …, 0.9}, per (model, option task), maximising dev strict macro-F1; ties go to the larger τ. Applied to test.
  - **(c) Temperature T** per (model, binary task), minimising the dev negative log-likelihood of σ(s/T) against hard gold. Used only for test ECE, Brier and soft metrics.
  - **(d) Platt scaling** σ(a·s + b) fitted on dev. Reported as an alternative decision rule.
  - **(e) EXIST dev evaluation** (3.2).

### 3.2 EXIST dev evaluation (LLMs only)
- **Gold:** the official dev golds (`data/EXIST/evaluation/golds/EXIST2025_dev_task{1_1,1_2,1_3}_gold_{hard,soft}.json`), restricted to the English ids of `processed/dev/exist_dev.parquet`.
  - Hard: n = 444 for 1.1, 400 for 1.2, 444 for 1.3.
  - Soft: n = 489.
- **Metrics:**
  - ICM, ICM-Norm, ICM-Soft and ICM-Soft-Norm via PyEvALL 0.2.11. Hierarchy for 1.2 `{"YES": [DIRECT, REPORTED, JUDGEMENTAL], "NO": []}`, for 1.3 `{"YES": [5 categories], "NO": []}`.
  - F1 as computed by PyEvALL.
  - Our strict macro-F1, and positive-class F1 for 1.1, with bootstrap CIs (B = 2,000, seed 42).
  - ICM values are point estimates (one PyEvALL call takes about 0.5 s, so bootstrapping them for 15 models is infeasible), with the official majority-class baseline beside them.
  - Prediction files use the `omit` and `answer-forced` variants and the 1.3 decoding of 2.9.
- **Reporting:** dev-split results, labelled as such, LLM-only (encoders select on dev). No "model X beats Y" claims, given the MDE at n ≈ 400–444. Contamination caveat: the EXIST 2025 tweets are probably the EXIST 2023 corpus (to be confirmed).
- **Note:** the PyEvALL install pins numpy 1.26 / pandas 2.2. It is installed with `--no-deps`, plus `tabulate` and `jsbeautifier`. Checked on 2026-10-23 against the official dev majority-class baseline files with the repo's numpy 2.5 and pandas 3.0.

### 3.3 Format check, label-free (`experiment: format-check`)
- **Sample:** 300 test posts per multiclass task (EDOS B, EDOS C, EXIST 1.2), seeded random (seed 42), label-blind.
- **Backend `text`:** greedy generation, up to 32 new tokens, with the v1 option prompt and no grammar.
- **Fixed parser** (`src/model/text.py`), applied in this order after removing the markup characters `*`, `#` and `` ` ``:
  1. the reply starts with an option letter A–K that is followed by `)`, `.`, `:`, `]`, a line end or the end of the reply, optionally inside "(" ")". This rule exists so a reply such as "A post that…" is not read as letter A;
  2. a phrase such as "answer/option/category is X" or "answer: X", with X an upper-case letter A–K;
  3. exactly one class name appears in the reply as whole words (case-insensitive);
  4. otherwise the reply is unparseable.

  Yes/No replies (self-explanations, 3.8): the first word, after markup is removed, is "yes" or "no" (case-insensitive); the rest of the reply is the rationale.
- **Report:**
  - the mismatch rate between the parsed letter and the first-token argmax letter
  - the unparseable rate
  - the parsed-answer rate among first-token non-answers.

### 3.4 Answer prefix v2 (`experiment: answer-prefix`)
- **Templates:** `src/prompts/templates/{edos/task_b,edos/task_c,exist/task_1_2}.v2.yaml` are v1 plus a log-prob prefill `"Answer: "` after the assistant turn starts. Separate files keep the v1 sha256 unchanged.
- **Runs:** all test posts, scored like the primary run (strict, τ = 0.5).
- **Report:** the non-answer rate and macro-F1 for v1 vs v2.

### 3.5 Option-order control (`experiment: option-order`)
- **Cyclic shifts:** `--option-perm k` moves class k to letter A (letters stay A.., classes rotate). EDOS B and EXIST 1.2 run with k = 1, 2, 3 on all test posts; k = 0 is the frozen run.
- **Label-free report:**
  - permutation consistency: the share of posts with the same predicted class under all 4 orders
  - the letter-choice distribution.
- **PH:** macro-F1 per order.
- **EDOS C, PriDe** (Zheng et al., ICLR 2024):
  - Estimate the letter prior on 200 seeded test posts (5%) under all 11 cyclic shifts.
  - Debias the other posts' frozen v1 scores: q(class) ∝ q(letter)/prior(letter).
  - Report debiased macro-F1 on the 970 gold-sexist posts minus the 200 prior posts (PH).

### 3.6 Five-prompt study PV-1 (`experiment: prompt5`)
- **Templates:**
  - T0 = the frozen v1 runs (reused, not rerun)
  - T1, T2 = rewordings of the system and question sentences
  - T3, T4 = format variants: post before the definition / no triple quotes / different answer instruction.
- **Fixed parts:** definition text, label names and canonical option order are identical in all five. Files `src/prompts/templates/{dataset}/{task}.T{1-4}.yaml`.
- **Freeze:** their sha256s go in `configs/prompt_variants.lock` before any run. `run_inference.py` refuses a variant whose file does not match the lock.
- **Runs:** all 7 tasks, test split, EDOS B/C on all 4,000 posts.
- **Report:**
  - per model: macro-F1 per template, mean ± SD over T0–T4, min and max
  - Kendall's W across templates; each model's rank range
  - prompt-averaged macro-F1 with a two-way (pigeonhole) bootstrap CI: posts and templates resampled independently, B = 2,000, seed 42
  - a variance decomposition of the model × template macro-F1 matrix: model, template and residual shares.

### 3.7 Encoder baselines (`experiment: encoders`)
- **Models:** `microsoft/deberta-v3-base` and `FacebookAI/roberta-base`, with revisions pinned at the first run and logged.
- **Tasks:** CMSB binary, EDOS A, EDOS B and EDOS C. B/C are trained on gold-sexist train posts and scored on the 970 gold-sexist test posts, as for the LLMs. EXIST is not scored (test unlabelled; dev is used for selection).
- **Fixed hyperparameters:**
  - AdamW, lr 2e-5, weight decay 0.01, 6% linear warm-up
  - batch 16, max length 128 tokens
  - 4 epochs for CMSB/EDOS A, 6 for EDOS B/C
  - bf16 on GPU.
- **Selection:** the checkpoint with the best dev macro-F1, evaluated each epoch.
- **Decision:** argmax. A dev-tuned threshold for binary tasks goes in a secondary row.
- **Seeds:** 13, 42, 87.
- **Report:** mean ± std, min–max and every seed's test macro-F1, plus each seed's bootstrap CI. Encoders are a reference point for the LLM numbers; no "improvement" claims (different paradigm, one train/dev split).

### 3.8 Explainability (`experiment: xai`, notebook 40)
- **(model, task) pairs:**
  - CMSB with Granite-4.1-8B (best primary CMSB model)
  - EDOS A with Gemma-4-E4B (best primary EDOS A model)
  - EXIST 1.1 dev with the best dev model by positive-class F1 (if it is one of the two above, the next best)
  - Gemma-4-E4B also on CMSB and EXIST 1.1 dev, for cross-task comparison.
- **Sample:** per pair, 25 seeded posts from each outcome cell TP/FP/FN/TN (all posts if a cell has fewer), selected in `99`.
  - Notebook 40 receives only a shuffled id list, so per-id outcomes (test labels) stay in `99`.
  - The analysis by outcome joins the attributions back in `99`.
- **Local attributions:**
  - SHAP Partition on log-odds, max_evals 300, mask "..."
  - LIME, 500 samples, seed 42; on a 20-post subset per pair, also seeds 13 and 87 for stability
  - SHAP–LIME agreement: Spearman ρ of word weights, and top-3 overlap.
- **Faithfulness** (ERASER):
  - comprehensiveness and sufficiency at k = 1, 3, 5 top SHAP words, against 5 random word sets of the same size (seed 42)
  - computed with "..." masking and with deletion (out-of-distribution check).
- **Plausibility:** for CMSB originals predicted sexist that have modifications, the words changed by the modification (word-level diff) are the gold rationale. Report precision@k of top SHAP words, k = number of changed words.
- **Global word effects:** mean SHAP by lexicon class (gender term, gendered slur, profanity, other) per pair. Tables of top words show slurs masked (e.g. "b\*tch").
- **Counterfactual tests** on all scored posts of the pair (label-free in notebook 40; split by gold class in `99`):
  - gender swap (lexicon map): Δs and the decision flip rate
  - removal of gendered slurs and profanity: Δs and the flip rate among posts predicted Yes.
- **Self-explanations:**
  - Backend `text`. The question is followed by "Answer Yes or No, then give one sentence explaining why." Greedy, up to 64 tokens.
  - Run on the FN and FP posts of the sample.
  - **Coding:** each rationale is coded into one of `{targets_women_or_gender, not_targeting_women, offensive_language, stereotype_or_inequality, reported_or_critical, humour_or_sarcasm, needs_context, opinion_not_hate, uninformative, other}`. The coder is Qwen3.5-4B, scoring lettered options with the log-prob scorer and a fixed coding prompt.
  - **Human check:** a seeded 20% subset is coded by a human; Cohen's κ is reported.
  - Self-explanations are reported as model claims, **not** faithful explanations (Turpin et al. 2023), and contrasted with the attribution and counterfactual evidence.

## 4. What is not in this addendum
These are not run unless a later addendum adds them: few-shot runs, the fp32 replication, contamination probes, HateCheck, guard models, persona prompting and thinking runs.
