# 10 · Evaluation review: is the current protocol the right one?

Written 2026-10-23, at the user's request. It reviews the frozen protocol in [06-final-evaluation.md](06-final-evaluation.md) (`notebooks/99_final_eval.ipynb`) against the literature. No code changed, nothing was run, and no test labels were read (only parquet column names were checked).

**Update 2026-10-23:** the 10-agent review in [11-osnem-roadmap.md](11-osnem-roadmap.md) §C corrects items 1, 2b, 3, 4, 6 and 8 and resolves several `TODO verify` items. Most importantly, cross-entropy should not be the primary soft metric; use L1/Brier instead. Read §C before acting on this page.

**Integrity constraint.** Test labels were read on 2026-10-20 under the frozen protocol. Everything below that is new is **post-hoc**: it can be reported as a secondary or exploratory analysis and must never replace or re-select the primary numbers. Items that need new runs (prompts, dev thresholds) need a new protocol entry in `docs/experiment-log.md` **before** scoring.

## Verdict

The primary hard-label protocol is sound and matches the official setups. Macro-F1 on EDOS A/B/C, B/C on gold-sexist posts, and ICM via EvALL for EXIST are what the shared tasks use. The statistics (paired bootstrap, Holm, strict refusals) are well done.

It is **not** the most informative evaluation possible, mainly for three reasons:
1. **One prompt per task.** The CIs ignore prompt variance, which the literature shows can change model rankings.
2. **A fixed 0.5 threshold on uncalibrated scores.** Macro-F1 then mixes ranking ability with a Yes-bias.
3. **Disagreement-aware (soft) evaluation is missing for CMSB and EDOS,** although the soft gold is in the test files.

## Keep as is

| Element | Why it is right | Source |
|---|---|---|
| EDOS B/C scored on gold-sexist posts | Official setting: "Only sexist instances are included in Task B and C". Macro-F1 for B/C over sexist posts only (organiser post on the Codalab forum, seen in a search snippet) | Kirk et al. 2023, [arXiv 2303.04222](https://arxiv.org/pdf/2303.04222) (quoted in `docs/report/zero_shot_sexism_llms.tex`); [Codalab forum](https://codalab.lisn.upsaclay.fr/forums/7084/1167) |
| Macro-F1 as primary for EDOS | Official metric of all three subtasks | same |
| EXIST via EvALL, ICM / ICM-soft, hard-hard and soft-soft | Official EXIST 2025 protocol (learning with disagreement) | Plaza et al. 2025, [CEUR 4038 paper 135](https://ceur-ws.org/Vol-4038/paper_135.pdf); ICM: Amigó & Delgado 2022, [ACL](https://aclanthology.org/2022.acl-long.399) |
| Paired bootstrap on a shared resample matrix, Holm over 14 comparisons | Standard, recommended practice for comparing systems on one test set | Dror et al. 2018, [ACL P18-1128](https://aclanthology.org/P18-1128/) |
| Strict scoring, refusal rate per label, τ sensitivity grid | Required by `.claude/rules/research-integrity.md`; shows format failures instead of hiding them | — |

**TODO verify:**
- The exact ICM / ICM-soft definitions and how the 1.2/1.3 hierarchy enters them. The EXIST 2025 overview PDF could not be machine-read in this session; check §Evaluation of the overview or the local `data/EXIST/EXIST_2025_Lab_Guidelines.V0.4.pdf`.
- The metric Samory et al. 2021 report for CMSB.

## Gaps, ranked by impact

Effort key: **L** = label-free on test or computable from existing `predictions.jsonl` (CPU); **G** = needs new GPU runs. Status: **2nd** = post-hoc secondary analysis; **new** = needs a new protocol written before scoring.

### 1. Prompt variance is not measured (G, new): biggest threat to the ranking claims
- **Issue.** One template per task (`src/prompts/templates/`). The bootstrap CIs only cover the sampling of posts.
- **Evidence from the literature:**
  - Sclar et al. (ICLR 2024, [arXiv 2310.11324](https://arxiv.org/abs/2310.11324v1)) find up to 76 accuracy points of spread from meaning-preserving format changes (LLaMA-2-13B, few-shot). They recommend reporting a range over plausible formats.
  - Mizrahi et al. (TACL 2024, [2024.tacl-1.52](https://aclanthology.org/2024.tacl-1.52)) find that different instruction paraphrases change both absolute scores and the **relative ranking** of 20 LLMs.
  - Zhao et al. (ICML 2021, [PMLR v139](https://proceedings.mlr.press/v139/zhao21c.html)) show that the choice and order of few-shot examples alone swing accuracy from near chance to near state of the art.
- **Recommendation.**
  - Use 3–5 paraphrased templates per task, written and frozen before any run. The few-shot sets from 2026-10-23 add more variants (repeat with 2–3 shot seeds).
  - Report mean ± SD over prompts per model, the rank of each model per prompt, and the best-minus-worst spread.
  - For model comparisons, use a two-level bootstrap (resample prompts, then posts), or report the per-prompt paired test.
- **Effect on the paper.** Claims such as "Granite-4.1-8B is best on CMSB" would rest on a prompt-robust ranking, not on one template.

### 2. Fixed 0.5 threshold on uncalibrated, Yes-biased scores (L, 2nd; dev thresholds G, new)
- **Issue.** Decisions are P(Yes)/mass ≥ 0.5, with no tuning.
- **Evidence in the repo** (`docs/06-final-evaluation.md`, `results/final_eval/tables/positive_rate.csv`, `calibration.csv`):
  - 14 of 15 models over-predict "sexist" on CMSB, and all 15 do on EDOS A (predicted rate 0.26–0.76 vs gold 0.24).
  - ECE ranges from 0.03 to 0.33 on CMSB and from 0.09 to 0.51 on EDOS A.
  - So macro-F1 at 0.5 partly measures each model's prior bias, not how well it separates sexist from non-sexist posts.
- **Recommendations** (all secondary):
  - **(a) Threshold-free discrimination: AUROC and AUPRC** from the stored scores, with bootstrap CIs and paired tests on the same resample matrix. Report both. McDermott et al. (NeurIPS 2024, [arXiv 2401.06091](https://arxiv.org/abs/2401.06091v4)) show AUPRC is *not* generally better than AUROC under imbalance and can favour high-prevalence subgroups.
  - **(b) Label-free bias correction**, with no test labels used:
    - *Contextual calibration*: score a content-free input ("N/A") and rescale (Zhao et al. 2021, above).
    - *Batch calibration*: estimate the bias from the mean score over the unlabelled test batch (Zhou et al., ICLR 2024, [arXiv 2309.17249](https://arxiv.org/pdf/2309.17249)).
    - A *prior-matched threshold* that sets the predicted positive rate to the **train** prevalence.
    - Each is a fixed rule, stated before scoring.
  - **(c) Dev-tuned per-label thresholds**, which `CLAUDE.md` originally planned. This would reverse the 2026-10-20 "test only" decision, so it is **the user's call**.

### 3. No disagreement-aware evaluation for CMSB and EDOS (L, 2nd)
- **Issue.** EXIST gets a soft-soft view through EvALL, but CMSB and EDOS are scored on hard labels only.
- **Evidence.**
  - The test parquets already hold the soft gold: `sexist_soft` (CMSB, 5 annotators) and `sexist_soft`, `category_soft`, `vector_soft` (EDOS, 3 annotators). Checked from the column names only.
  - LLM log-prob scores are already probability distributions over the answers.
  - The learning-with-disagreement literature argues that hard and soft evaluation can rank systems differently and that soft evaluation is needed for subjective tasks:
    - Uma et al. 2021, [JAIR 72](https://jair.org/index.php/jair/article/view/12752)
    - LeWiDi at SemEval-2023, which ranks by cross-entropy (Leonardelli et al., [arXiv 2304.14803](https://arxiv.org/pdf/2304.14803)).
- **Recommendation.**
  - Add soft metrics against the annotator distribution: cross-entropy (primary soft metric, as in LeWiDi), Brier, and Jensen–Shannon divergence. This covers CMSB, EDOS A, and EDOS B/C on gold-sexist posts.
  - Normalise each dataset's soft definition first, from votes and `n_annot`. `data/README.md` warns the soft columns mean different things per dataset; for example, CMSB's soft value uses the Samory rule.
  - Then every dataset has a hard view and a soft view.

### 4. Abstention is scored at one cutoff only (L, 2nd)
- **Issue.** A label is refused when mass < 0.5 and then counted as wrong. The τ grid {none, 0.1, 0.5, 0.9} is a coarse sweep. The all-15 "fair subset" is **empty for EDOS C** and has 865 of 970 posts for EDOS B (06 doc).
- **Recommendation.**
  - Treat answer mass as a confidence score. Report a **risk–coverage curve** and its area (AURC; selective classification, Geifman & El-Yaniv 2017, [arXiv 1705.08500](https://arxiv.org/abs/1705.08500)) per model on EDOS B/C and EXIST 1.2.
  - Replace the all-15 intersection with **pairwise fair subsets**: model vs best, on the posts both answered.
- The strict number stays primary.

### 5. "Tied with best" is read from non-significance (L, wording only)
- **Issue.** `tied_with_best` = Holm-adjusted p ≥ 0.05. A non-significant difference is absence of evidence, not evidence of equality. Many NLP comparisons are underpowered (Card et al., EMNLP 2020, [arXiv 2010.06595](https://arxiv.org/pdf/2010.06595)), and EDOS B/C have only 970 scored posts.
- **Recommendation.**
  - Make the Δ-vs-best CI the main evidence. Word ties as "not separable at this test size", not "equivalent".
  - Optionally report the minimum detectable difference.
  - A formal equivalence test (TOST) would need a margin fixed in advance, which is not possible now, so it is not recommended post hoc.

### 6. Calibration is measured with binned ECE only (L, 2nd)
- **Issue.** ECE with 10 equal-width bins depends on the binning, and the bf16 score grid makes it worse (`docs/09-known-issues.md`).
- **Recommendation.** Add the **Brier score** (a proper scoring rule, and the same quantity as item 3's soft Brier against hard gold). Optionally add adaptive-bin ECE. Keep the reliability plots.

### 7. CMSB's adversarial design is unused (L, 2nd)
- **Issue.** CMSB was built with minimally edited adversarial versions of sexist posts (`of_id`) to test for over-reliance on lexical markers (Samory et al., ICWSM 2021, [arXiv 2004.12764](https://arxiv.org/pdf/2004.12764)). Only the overall score is reported.
- **Recommendation.**
  - Score originals and modifications separately.
  - Report a pair flip rate: an original predicted sexist whose modification is still predicted sexist, which shows lexical shortcutting.
  - Check first how many original–modification pairs are among the 1,126 scored test posts (not checked: it reads `of_id`, which reveals labels, so it belongs in `99`).
- **Optional external test.** HateCheck (Röttger et al., ACL 2021, [repo](https://github.com/paul-rottger/hate-functional-tests)) has functional tests with **women** as one of 7 target groups (verified in the repo README). It probes negation, counter-speech and quoted hate, which matches EXIST's REPORTED/JUDGEMENTAL classes. It needs new (cheap) GPU runs.

### 8. Possible test-set contamination is not discussed (G, 2nd)
- **Issue.** CMSB (2021) and EDOS (2023) are public. The 15 models (frozen 2026-10-20) were likely trained after the release, so zero-shot numbers may partly reflect memorisation.
- **Recommendation.**
  - Run a label-free probe: a guided-instruction completion test (Golchin & Surdeanu, [arXiv 2308.08493](https://arxiv.org/abs/2308.08493v1)) on a sample of test posts, or a per-post log-likelihood comparison of test vs train posts.
  - At minimum, add a limitation paragraph.

### 9. Minor
- **EDOS C per-class support is small:** 14 posts for the rarest vector (3.4), per `zero_shot_sexism_llms.tex`. Report per-class n and per-class F1 CIs next to `per_class_f1_task_c.csv`.
- **MCC for binary tasks:** optional secondary single-number summary. Macro-F1 stays primary because it is the official metric.
- **EXIST:** English-only EvALL submission, and handling of omitted items unconfirmed. Already in `09-known-issues.md`; keep both `strict` and `nocut` files.

## Suggested order

| # | Addition | Effort | Status | Uses test labels |
|---|---|---|---|---|
| 1 | AUROC/AUPRC + Brier (items 2a, 6) | L | 2nd | yes, in `99` only |
| 2 | Soft metrics CMSB/EDOS (3) | L | 2nd | yes, in `99` only |
| 3 | Δ-CI wording + pairwise fair subsets + risk–coverage (4, 5) | L | 2nd | yes, in `99` only |
| 4 | CMSB original vs modification analysis (7) | L | 2nd | yes, in `99` only |
| 5 | Label-free calibration rules (2b) | L (+1 forward pass for contextual) | 2nd, rule fixed in log before scoring | no for the rule, yes for the score |
| 6 | Multi-prompt + few-shot seeds (1) | G | new protocol | yes, after freezing |
| 7 | Contamination probe (8), HateCheck (7) | G | 2nd | no / HateCheck gold |
| 8 | Dev-tuned thresholds (2c) | G | new; reverses 2026-10-20 decision | dev only |

Items 1–4 need only CPU and the existing predictions. Item 6 changes most what the paper can claim.

## Sources (accessed 2026-10-23)
- Kirk et al. 2023, SemEval-2023 Task 10 (EDOS): <https://arxiv.org/pdf/2303.04222>; organiser forum post: <https://codalab.lisn.upsaclay.fr/forums/7084/1167>
- Plaza et al. 2025, EXIST 2025 overview: <https://ceur-ws.org/Vol-4038/paper_135.pdf>; EXIST 2023 overview: <https://doi.org/10.1007/978-3-031-42448-9_23>
- Amigó & Delgado 2022, Evaluating extreme hierarchical multi-label classification (ICM): <https://aclanthology.org/2022.acl-long.399>
- Samory et al. 2021, "Call me sexist, but…": <https://arxiv.org/pdf/2004.12764>
- Uma et al. 2021, Learning from disagreement: a survey: <https://jair.org/index.php/jair/article/view/12752>
- Leonardelli et al. 2023, SemEval-2023 Task 11 LeWiDi: <https://arxiv.org/pdf/2304.14803>
- Sclar et al. 2024, Quantifying LMs' sensitivity to spurious features in prompt design: <https://arxiv.org/abs/2310.11324v1>
- Mizrahi et al. 2024, State of what art? Multi-prompt LLM evaluation: <https://aclanthology.org/2024.tacl-1.52>
- Zhao et al. 2021, Calibrate before use: <https://proceedings.mlr.press/v139/zhao21c.html>
- Zhou et al. 2024, Batch calibration: <https://arxiv.org/pdf/2309.17249>
- McDermott et al. 2024, A closer look at AUROC and AUPRC under class imbalance: <https://arxiv.org/abs/2401.06091v4>
- Dror et al. 2018, Hitchhiker's guide to significance testing in NLP: <https://aclanthology.org/P18-1128/>
- Card et al. 2020, With little power comes great responsibility: <https://arxiv.org/pdf/2010.06595>
- Geifman & El-Yaniv 2017, Selective classification for deep neural networks: <https://arxiv.org/abs/1705.08500>
- Röttger et al. 2021, HateCheck: <https://aclanthology.org/2021.acl-long.4>, <https://github.com/paul-rottger/hate-functional-tests>
- Golchin & Surdeanu, Time travel in LLMs (contamination): <https://arxiv.org/abs/2308.08493v1>

Most papers were checked from abstracts, proceedings pages and search snippets, not full text. Verify specific numbers (e.g. "76 points") against the PDF before they go into the paper.
