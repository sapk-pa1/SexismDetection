# 11 · Roadmap to a top-tier OSNEM submission (2026-10-23)

**Target venue and timing.** The Elsevier journal *Online Social Networks and Media* (OSNEM), submission mid-November 2026 (about 3 weeks). Items are ranked deadline-feasible first; bigger ideas are under "Later".

**How this roadmap was made.**
- Ten research agents, run 2026-10-23, read the repo and the literature. Nine worked in parallel on focused topics, then a "reviewer 2" critic audited, deduplicated and spot-checked them. The orchestrator wrote this synthesis.
- Per-agent findings, sources and progress logs were in the session scratchpad (`…/scratchpad/research/NN_*/`). The scratchpad is temporary, so the key sources are copied below.
- Nothing in the repo was changed or run.
- **Test labels:** no agent read test label columns, with one exception (§ Integrity note).

**Builds on.** [10-evaluation-review.md](10-evaluation-review.md). Section C below corrects parts of it.

**Implementation status (2026-10-23, after the user's decisions D1 standalone, D2 dev allowed, all GPU items in, EXIST kept on dev).** Rules for every item are in [12-protocol-addendum.md](12-protocol-addendum.md). The code, notebooks 30/40 and the `99` post-hoc section are done and sanity-checked on CPU; nothing is run yet.
- **Fixed in code:**
  - E1: simultaneous intervals and wording.
  - E2: variant names. The refused → NO variant was dropped, because it would fill a refusal with a class.
  - E4.
  - E6: tie flag.
  - E9: docs.
  - E11.
- **Not fixed: E10.** Few-shot is outside addendum A1, so the counterbalanced shot sets were not built. The existing `configs/fewshot/*_k1_s42.yaml` sets must not be run as they are.
- **Implemented, waiting for runs:** P0–P2, P4, P6–P11, P15. P13 (fp32) and P14 (few-shot) are not in the addendum.
- **Still writing work:** P3, P5, P12 (paper and front matter).

**Abbreviations.**
- *Integrity status:* **PU** = primary unchanged; **PH** = post-hoc secondary (test labels were read 2026-10-20); **PR** = a new protocol pre-registered before scoring.
- *Effort:* **L** = CPU on existing predictions; **G** = new GPU runs; **W** = writing only.

## A. Verdict
As a reviewer would see the current draft (`docs/report/zero_shot_sexism_llms.tex`): **major revision at best**. Rejection is likely if two gaps both stay open:
- no same-split supervised reference (R3);
- EXIST unscored, with a planned scoring that would be wrong (R4).

The frozen hard-label protocol itself is sound. The gaps are in what is measured around it, in how some numbers are presented, and in OSNEM scope fit.

Most must-do fixes are writing and CPU work: about 12–13 person-days and about 4 A100 hours. A prompt-robustness study adds about 20 GPU hours.

## B. Errors and inconsistencies found in the current repo or paper (fix regardless of scope)
| # | Problem | Evidence | Fix | Status |
|---|---|---|---|---|
| E1 | Two rows are marked "not separable from best" while their 95% Δ interval excludes 0: CMSB Jamba2-3B Δ −.029 [−.056, −.003], Holm p .141; EDOS A Qwen3.5-4B Δ −.018 [−.032, −.003], Holm p .056. The interval is per comparison, while the dagger uses the Holm family-wise rule. | `results/final_eval/tables/main_results_full.csv` (checked by the orchestrator) | Add simultaneous (Bonferroni .05/14) Δ intervals, or a footnote; print p = 0 as "< 1e-4" | PU (presentation) |
| E2 | The EvALL "strict" files *omit* refused items. EvALL scores an omitted item as "no category", which beats a wrong answer, so this variant is lenient. | Agent 01: PyEvALL v0.2.11 source read; tested on EXIST **dev** gold with synthetic predictions. On 1.1, dropping 20% of items gives ICM-Norm 0.80, answering them wrong gives 0.70 | Rename the variants ("omit", "answer-forced"); add a truly strict variant (refused → `NO`) | PU (EXIST still unscored) |
| E3 | English-only predictions against the bilingual gold score badly even when perfect (ICM-Norm 0.47, F1 0.64). The official EXIST 2025 leaderboards are es+en combined. | Agent 01 (dev test); agent 07 | Ask the organisers whether an English-only test case exists **before** any upload; if none by about Nov 6, take EXIST out of the headline | PU |
| E4 | The EXIST 1.3 soft `NO` is computed as Π(1 − P(Yes)). In the official gold, 1.3 NO equals 1.1 NO (dev: 1,038 of 1,038 items). | Agent 01/05; `src/utils/evall.py` | Use the model's own 1.1 P(No); gate 1.3 categories on 1.1 (hierarchical decoding). Pre-register now, while EXIST is unscored | PU if done before scoring |
| E5 | The ≥ 80% EDOS C "refusals" (Aya, SmolLM3, Granite) are first-token format non-answers ("The", "None", "\*\*"). The log-prob path never applies `template.prefill`. In multiclass strict scoring, a refusal costs recall but adds no false positive. | Agent 02/04; `src/model/logprob.py`, `src/prompts/builder.py`; Wang et al. 2024 (first-token vs text answers disagree in > 60% of cases) | Say "format non-answer"; label τ = none "answer-forced"; label-free top-token taxonomy; pre-registered "Answer: " prefill variant | PU (wording) / PR (prefill) |
| E6 | Exact ties in letter probabilities go to the earlier class (`max()` in schema order): 534 of 60,000 EDOS C predictions (169 of 4,000 for Llama). bf16 logits step Yes/No log-odds by about 0.125. | Agent 02 (stored predictions, label-free) | Disclose the tie rule and tie counts; fp32 last-position rerun under "could" | PU |
| E7 | CMSB test is not all social-media posts: 16% are psychological-scale items. About 40% are crowd-written adversarial edits. Agent 08, from train/dev: about 58% of non-sexist posts are edits. | Agent 09 (see Integrity note); agent 08 (train/dev aggregates) | Call CMSB examples "items"; contrast-set analysis (P11b); add a composition paragraph | PU (wording) / PH (analysis) |
| E8 | The EXIST "2025" tweet split sizes (6,920 / 1,038 / 2,076) match EXIST 2023 exactly, so the text has probably been public since 2023. The Limitations section lists only CMSB and EDOS as possibly seen in pre-training. | Agent 08; confirmed for 2023 by agent 10; 2025 reuse **unverified** | Confirm in the EXIST 2025 overview. Extend the contamination limitation to EXIST. Check whether the 2023 test gold was released; if so, tex l.184 is wrong | PU |
| E9 | Docs disagree about scope: CLAUDE.md, README and `evall_team "HEFT"` present HEFT, while the report removed it; CLAUDE.md says Qwen3.5-4B runs with thinking on and off (never run); CLAUDE.md's "multi-label" framing doesn't match the single-label EDOS B/C. | Agents 03, 07, 10 | Settle the HEFT scope (D1) and fix the docs | PU |
| E10 | The 2026-10-23 few-shot sets (`configs/fewshot/`) end every binary set with a No shot, and there is only one shot seed. Since every model over-flags, a few-shot "gain" could be recency bias. | Agent 03; noted in the log at creation | Before any few-shot run: counterbalance shot order, use 3 shot seeds, and a compact multiclass format | PR (no few-shot result seen yet) |
| E11 | SmolLM3's chat template injects the run date into every prompt; the stored template sha doesn't capture it. | Agent 02 | Store a sha of the rendered prompt; disclose | PU |
| E12 | Unverifiable citations in the draft: the Riahi Samani zero-shot ranges (tex l.135), and "EXIST 2025 listed in EvALL" (l.184). | Agents 07, 10 | Check the PDF by hand or remove | PU |

## C. Corrections to [10-evaluation-review.md](10-evaluation-review.md)
- **Item 3 (soft metrics):** do **not** use cross-entropy as the primary soft metric.
  - On CMSB train, the hard gold used as a prediction gets CE 1.32 vs 0.63 for a constant prior, and its CE ranges from 0.66 to 1.98 depending only on the clipping ε. Our scores are often saturated.
  - LeWiDi-2025 dropped CE for Manhattan distance (agent 05; confirmed by 10).
  - Use **L1 and Brier against annotator shares** as primary, with CE in the appendix. EDOS B/C category shares must be renormalised over sexist votes.
- **Item 2b (calibration):** plain contextual and batch calibration push toward *uniform* classes.
  - Label-free check: they move predicted positive rates toward 0.5 and away from gold. For example, SmolLM3 on EDOS A goes from 0.259 to 0.589 against a gold rate of 0.24 (agent 04).
  - Use the **prior-aware** form (p · π_train / p̂_model).
  - Train and test prevalence are nearly identical (CMSB 0.320 / 0.321; EDOS A 0.243 / 0.243), so a train-prior threshold is near-oracle. Present it as the precision–recall break-even point (ranking quality), not as a deployable rule.
- **Item 4 (risk–coverage):** mass ≈ 1 on all binary tasks, so use |log-odds| or the maximum renormalised probability as the confidence score there, and mass only on option tasks.
- **Item 8 (contamination):** a train-vs-test likelihood probe cannot detect anything, because the splits were released together and membership inference is near chance.
  - Use ID-string counts in the infini-gram indexes (label-free, no text sent).
  - Use a DCQ-style quiz scored by the existing letter scorer.
  - Add a limitation paragraph.
- **Item 1 (prompt variance):** prompts and posts are *crossed*, so use a pigeonhole (two-way) bootstrap or a mixed model, with ≥ 5 templates; 3 templates give only about 10 distinct resamples.
  - Run all 15 models: picking the top models by test score would be selection on test.
  - Run EDOS B/C on all 4,000 posts: restricting inference to gold-sexist posts uses labels to choose inputs.
- **TODOs resolved:**
  - Samory et al. report macro-F1, on balanced random 70/30 splits, so there is no comparable CMSB number.
  - EXIST officially uses ICM / ICM-soft, plus ICM-Norm and F1 (agent 01, from the overview).
  - HateCheck has a women target group.
- **Item 6:** the bootstrap p (`src/utils/stats.py`) is an interval inversion, not a simulated null. Describe it as such, and optionally add an approximate-randomization check.

## D. Decisions only the user can make (needed by about Mon 2026-10-26)
1. **D1 HEFT scope.** Submit the zero-shot paper standalone (recommended by 07, 09 and 10) or wait for HEFT. If standalone, update CLAUDE.md, README, docs and the EvALL team name.
2. **D2 Dev runs.**
   - Allow label-blind dev inference (about 2 GPU h) as a pre-registered secondary. This would give dev thresholds and calibration, a local EXIST dev ICM, and an LLM row comparable to dev-tuned encoders.
   - Or keep the 2026-10-20 test-only decision.
   - The trade-off: the dev labels then also serve the LLM study. Recommended by 04 and 10: yes.
3. **D3** Add the PyEvALL package (`pip` rule in CLAUDE.md) for local ICM checks.
4. **D4** EXIST fallback: remove it from the headline if no English-only score arrives by about Nov 6 (recommended).
5. **D5** GPU budget: about 30 A100 h covers P7, P8, PV-1 and fp32; the minimum is P7b at about 1–2 h.
6. **D6** Train the DeBERTa-v3 / RoBERTa baselines before submission (3 seeds, about 6–10 GPU h, TODO measure).
7. **D7** Data deposit terms (IDs and per-post scores, no text), and whether the EXIST 2023 test gold is public.
8. **Model list:** no changes before submission (guard models and a frontier reference run go under Later). This is a recommendation; the decision is the user's.

## E. Priority list (impact per day, 3-week deadline)
| # | Item | Effort, GPU | Status | Tier |
|---|---|---|---|---|
| P0 | **Protocol addendum** committed *before* any new analysis. It lists every PH/PR item with metrics, ε, tie rules and bootstrap re-estimation. The paper gets a subsection "Analyses added after test labels were read (2026-10-20)". | W 0.5 d | enables PH/PR | must |
| P1 | **Statistics display**:<br>• E1 fix<br>• MDE column (about .035 CMSB, .024 EDOS A, .062 EDOS B, .046 EDOS C at 80% power; approximate, recompute from the bootstrap)<br>• "not separable at this test size" wording<br>• P(best) and rank intervals<br>• size–F1 Spearman with CIs and Holm (drop the "size helps on EDOS B" reading: ρ .52 [−.01, .82], Holm p ≈ .18)<br>• confirmatory vs exploratory test families<br>• cross-task Kendall τ (EDOS A vs B τ = 0.03) | L/W 1.5 d | PU / PH | must |
| P2 | **EXIST rescue**: (a) email the organisers now; (b) pre-register the export fixes E2 and E4; (c) local PyEvALL check (D3); (d) the Nov 6 fallback (D4). | W 0.5 + L 1.5 d | PU | must |
| P3 | **Positioning**:<br>• EDOS reference ladder: majority, organiser baselines, median team (A .832 / B .619 / C .423), top<br>• verified frontier zero-shot rows: GPT-5.2 .731 / .454 / .338 (Alajmi & Pergola 2026); GPT5-mini 73.0 / 41.9 / 41.0 and Qwen3-4B 66.7 / 37.9 / 20.2 (Edman et al. 2026), labelled "different prompts/parsing, indicative"<br>• rewrite the contributions around local, label-free deployability<br>• about 10 new 2024–26 references incl. OSNEM papers and Zhang et al. 2024 (over-sensitivity)<br>• no "first" claims | W 1.5 d | PU | must |
| P4 | **Non-answer reframing** (E5): wording, answer-forced label, top-token taxonomy, multiclass strict asymmetry. | W/L 1 d | PU | must |
| P5 | **OSNEM fit**:<br>• provenance table (EDOS: Gab plus 81 manosphere subreddits, 2016–18; CMSB: scale items plus tweets 2008–19; EXIST: Twitter)<br>• moderation load: false flags per 1,000 posts, precision at the observed prevalence<br>• size, peak-GPU (2.6–18 GB from `complete.json`) and licence table<br>• fuller ethics section; Limitations before the Conclusion<br>• keywords, highlights, CRediT, competing interests, **Generative-AI declaration**<br>• `elsarticle`<br>• Zenodo DOI | W 2.5 + L 0.5 d | PU | must |
| P6 | **Threshold-free and soft metrics**: AUROC/AUPRC on log-odds, Brier, MCC, per-class F1 with n and CIs; L1/Brier against annotator shares; hard-vs-soft rank τ. | L 2 d | PH | must |
| P7 | **Is the non-answer a scoring artefact?** (b) Label-free generate-then-parse sample: 300 posts × 3 multiclass tasks × 15 models, mismatch rate. (a) "Answer: " prefill variant (new template version) on EDOS B/C and EXIST 1.2. | G+L 1.5 d, about 3–4.5 GPU h | PU (b) / PR (a) | must |
| P12 | **Doc and disclosure fixes**: E6, E8, E9, E11, E12; contamination limitation for all 3 datasets; dataset-year vs model-cutoff table; inference sanity appendix (13/15 tokenizers checked clean); verify the 6 abridged EDOS C definitions. | W 1.5 d | PU | must |
| P5b | **PV-1 prompt study**: 5 frozen templates (T0, 2 rewordings, 2 format variants), all 15 models, all 7 tasks. Report AvgP ± SD, MaxP, Kendall's W, rank ranges; pigeonhole bootstrap. | G+L 4 d, about 18–22 GPU h | PR | must if D5 allows, otherwise a stated limitation |
| P11 | **OSN angle "whose sexism do LLMs flag?"**:<br>• false positives split by annotator support (no annotator vs a minority called it sexist), with the P(flag) vs annotator-share curve<br>• EDOS expert-overridden posts<br>• CMSB contrast set: true-negative rate on edits vs originals, pair accuracy, within-pair log-odds gap<br>• uncertainty vs disagreement | L 3 d | PH | should (highest-novelty CPU item) |
| P9 | **Label-free threshold views**: train-prior log-odds row (break-even reading), prior-aware batch calibration next to plain, a recalibration-method table. | L 1.5 d | PH | should |
| P8 | **Option-order control**: cyclic permutations (EDOS B, EXIST 1.2), 5% PriDe on EDOS C; the consistency rate is label-free. | G+L 1.5 d, about 4.5 GPU h | PU / PH | should |
| P10 | **Supervised reference**: DeBERTa-v3-base and RoBERTa-base on each train split, dev-selected, 3 seeds; seed protocol frozen first. Gives CMSB its only anchor. | G 2.5–3 d, about 6–10 GPU h | PR | should (must if CMSB stays a headline); depends on D1, D6 |
| P13 | fp32 last-position replication and label-free flip rate. | G+L 1.5 d, about 5 GPU h | PU / PR | could |
| P14 | Few-shot k = 1, only after the E10 fix; binary tasks first. | G+L 3 d | PR | could |
| P15 | Hierarchical and end-to-end EDOS views; label-free cross-task coherence (C under its own B; EXIST 1.2/1.3 vs 1.1). | L 1–1.5 d | PH / PU | could |
| P16 | Dev sweep (needs D2). | G 1 d, about 2 GPU h | PR | could (high value if D2 = yes) |
| P17 | Contamination tiers 1–2; gender-word-lexicon and Perspective-toxicity reference rows (Samory: a gender-word rule alone reaches 73%); LLM as an extra annotator vs human leave-one-out agreement (0.896 CMSB / 0.912 EDOS on train). | L/G 2–3 d, about 2 GPU h | PH | could |

**Totals.**
- Must items without PV-1: about 12–13 person-days and about 4 GPU h.
- PV-1 adds about 4 days and about 20 GPU h (about one background day at `max_parallel` 7).
- Should items add about 8.5–9 days, so not all fit. In order: P11, P9, P8, then P10 if D1/D6 allow.

## F. Three-week schedule (critic's proposal, adjusted)
| Dates | Work |
|---|---|
| Oct 23–27 | Decisions D1–D7; email the EXIST organisers. Write and commit P0, the PV-1 templates and checklist, the prefill template, the permutation code and the EvALL export rules. |
| Oct 28–30 | GPU in the background: P7b then P7a, P8, P5b, and P10 if D6. CPU meanwhile: P1, P4, P6 in `99`. |
| Nov 2–6 | P11 and P9 (CPU, in `99`); PV-1 and permutation analysis. EXIST go/no-go by Nov 6. Writing: P3, P12. |
| Nov 9–11 | P5 writing (provenance, moderation load, ethics, front matter, `elsarticle`, highlights, GenAI declaration); Zenodo deposit. |
| Nov 12–13 | Consistency pass: every number traced to a `run_name` and a CSV, primary vs post-hoc labels checked, an internal reviewer-2 read; submit. Slack until Nov 16. |

**Risks.**
- EvALL silence: handled by the Nov 6 fallback.
- Colab A100 availability: PV-1 is the item to drop, stated as a limitation.
- Scope creep from the "could" items.

## G. Later / revision
- Guard models (Llama Guard, ShieldGemma) as a reference group. This would reverse the 2026-10-20 removal of Granite Guardian, so it is the user's call.
- HateCheck women subset (with the label mapping pre-registered).
- Explanation faithfulness (ERASER metrics; CMSB edit spans as rationales).
- Qwen3.5 thinking-then-score.
- kNN demonstrations and k-scaling.
- Class-name / PMI scoring.
- Full 11-way option cycling on EDOS C.
- Persona prompting: evidence is weak and prompt-sensitive.
- EXIST demographic analysis: on English train, female- vs male-majority decisions differ on 21.8% of tweets vs 22.3% for random splits, so there is no effect beyond chance at the binary level.
- Encoder cross-dataset transfer.
- The HEFT paper.

## Integrity note
- **Deviation.** Agent 09 read the raw CMSB `of_id` for **test** items to count the test composition, which the agent brief forbade, because `of_id` reveals labels (edits are almost always non-sexist).
  - Only aggregate counts were produced: 451 edits, and the per-source counts behind E7.
  - No metric was computed, and the frozen primary results from 2026-10-20 are unaffected.
  - Any CMSB contrast-set analysis (P11b) must be pre-specified in P0 and run in `99`, and the paper should state that the composition was inspected after the primary scoring.
- **Agent 01's EvALL check.** It ran PyEvALL from the scratchpad, with two missing dependencies stubbed, on EXIST **dev** gold and synthetic predictions only. Adding PyEvALL to the repo is decision D3.
- **Repo.** No agent changed it: `git status` matched before and after the sweep.

## Key sources (verified by the agents unless marked)
- Kirk et al. 2023, EDOS: <https://arxiv.org/pdf/2303.04222>
- Samory et al. 2021, CMSB: <https://ar5iv.labs.arxiv.org/html/2004.12764>
- EXIST 2025 overview, CEUR 4038 paper 135: <https://ceur-ws.org/Vol-4038/paper_135.pdf>. The evaluation section was extracted by agent 01.
- Alajmi & Pergola 2026, zero-shot EDOS: <https://arxiv.org/html/2512.23732>. Table 2 spot-checked by the critic; venue unconfirmed.
- Edman et al. 2026: <https://arxiv.org/html/2608.25605>. Tables 2/4 spot-checked by the critic.
- Tian et al. 2025, EXIST 2025 winner, LoRA Llama-3.1-8B: <https://arxiv.org/html/2507.10996v1>
- Wang et al. 2024, first-token vs text answers: <https://arxiv.org/abs/2402.14499>
- Zheng et al. ICLR 2024, MCQ selection bias / PriDe: <https://arxiv.org/abs/2309.03882>
- Wen et al. TACL 2025, abstention survey: <https://aclanthology.org/2025.tacl-1.26>
- Zhao et al. 2021, contextual calibration: <https://proceedings.mlr.press/v139/zhao21c.html>
- Fei et al. ACL 2023, domain-context calibration: <https://aclanthology.org/2023.acl-long.783/>
- Zhou et al. ICLR 2024, batch calibration: <https://arxiv.org/pdf/2309.17249>
- Leonardelli et al. 2025, LeWiDi-2025 (Manhattan distance instead of CE): <https://arxiv.org/html/2510.08460v2>
- Calderon et al. 2025, alternative-annotator test: <https://arxiv.org/abs/2501.10970>
- Sclar et al. ICLR 2024: <https://arxiv.org/abs/2310.11324v1>
- Mizrahi et al. TACL 2024: <https://aclanthology.org/2024.tacl-1.52>
- Card et al. EMNLP 2020: <https://arxiv.org/pdf/2010.06595>
- Miller 2024, adding error bars to evals: <https://arxiv.org/html/2411.00640>
- Dror et al. TACL 2017, replicability across datasets: <https://aclanthology.org/Q17-1033/>
- Ulmer et al. 2022, deep-significance / ASO: <https://ar5iv.labs.arxiv.org/html/2204.06815>
- Owen 2007, pigeonhole bootstrap: <https://arxiv.org/pdf/0712.1111> [snippet-only]
- Zhang et al. ACL 2024, over-sensitivity in implicit hate detection: <https://aclanthology.org/2024.acl-long.652/> [abstract]
- Golchin & Surdeanu, DCQ: <https://arxiv.org/abs/2311.06233> [snippet-only]
- infini-gram API: <https://infini-gram.readthedocs.io/en/latest/api.html>
- Röttger et al. 2021, HateCheck: <https://ar5iv.labs.arxiv.org/html/2012.15606>
- OSNEM Guide for Authors (2021 snapshot; the live page was blocked, so re-check it in a browser): <https://web.archive.org/web/20210115235832id_/https://www.elsevier.com/journals/online-social-networks-and-media/2468-6964/guide-for-authors>
- Elsevier generative-AI policy: <https://www.elsevier.com/about/policies-and-standards/the-use-of-generative-ai-and-ai-assisted-technologies-in-writing-for-elsevier>
- OSNEM papers to cite (abstracts): SWSR <https://doi.org/10.1016/j.osnem.2021.100182>; others listed by agent 09 (Yin & Zubiaga 2022, Enneifer 2025, Rodriguez-Vega 2026, Nudo 2025). Read the full texts before citing.

Numbers quoted from papers were spot-checked by the critic where marked. Agents read many papers only as abstracts or snippets, so check every number in the full text before it goes into the paper.
