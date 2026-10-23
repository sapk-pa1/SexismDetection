# 4 · LLM inference: models, schemas, prompts, backends, run folders

Code: `src/schemas/`, `src/prompts/`, `src/model/`, `src/utils/`, driven by `run_inference.py`. Model list: `configs/llms.yaml`.

All LLMs are used **zero-shot** with **next-token log-probability scoring**: no fine-tuning, no examples in the prompt, no sampling. This follows the research rule "score each label as a Yes/No log-probability, not free generation".

## 4.1 Model registry (`configs/llms.yaml`)

15 open-weight instruct checkpoints, **≤ 8B total parameters by the model-card figure**, frozen 2026-10-20 with commit shas resolved from the Hub that day.

| Short name | Repo | Params (card) | Licence | Gated | Notes |
|---|---|---|---|---|---|
| `qwen3.5-4b` | Qwen/Qwen3.5-4B | 4B (Hub 5B incl. vision) | apache-2.0 | no | thinking on by default; turned off for logprob |
| `gemma4-e4b` | google/gemma-4-E4B-it | E4B (effective 4B) | apache-2.0 | no | |
| `ministral3-3b` | mistralai/Ministral-3-3B-Instruct-2512 | 3.4B + 0.4B vision | apache-2.0 | no | FP8-only checkpoint: `fp8_dequantize: true` (A100 has no native FP8) |
| `olmo3-7b` | allenai/Olmo-3-7B-Instruct | 7B | apache-2.0 | no | needs transformers ≥ 4.57 |
| `llama3.1-8b` | meta-llama/Llama-3.1-8B-Instruct | 8B | llama3.1 | manual | |
| `phi4-mini` | microsoft/Phi-4-mini-instruct | 3.8B | mit | no | |
| `granite4.1-8b` | ibm-granite/granite-4.1-8b | 8B (Hub 9B) | apache-2.0 | no | first to swap under a strict 8.0B cutoff |
| `nemotron3-nano-4b` | nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | 3.97B | nvidia-nemotron-open-model | no | thinking on by default; turned off via `enable_thinking=False` |
| `apertus-8b` | swiss-ai/Apertus-8B-Instruct-2509 | 8B | apache-2.0 | no | |
| `smollm3-3b` | HuggingFaceTB/SmolLM3-3B | 3B | apache-2.0 | no | thinking on by default; turned off |
| `glm-edge-4b` | zai-org/glm-edge-4b-chat | 4B | glm-4 (custom) | no | |
| `falcon-h1-7b` | tiiuae/Falcon-H1-7B-Instruct | 7B (Hub 8B) | falcon-llm | no | hybrid Mamba; slow without `mamba-ssm` |
| `exaone4-1.2b` | LGAI-EXAONE/EXAONE-4.0-1.2B | 1.28B | exaone-1.2-nc (non-commercial) | no | thinking opt-in only |
| `aya-expanse-8b` | CohereLabs/aya-expanse-8b | 8B | cc-by-nc-4.0 | automatic | |
| `jamba2-3b` | ai21labs/AI21-Jamba2-3B | 3B | apache-2.0 | no | hybrid Transformer–Mamba; replaced Granite Guardian 3.3-8B on 2026-10-20 |

The pinned revision sha of each model is in `configs/llms.yaml` and in `results/final_eval/paper/models.tex`.

Fields per entry (`src.utils.registry.LLMSpec`; unknown keys fail, so a typo can't drop an option):
- `display_name` (figures and tables only; never in run configs), `repo_id`, `revision`, `params`, `licence`, `gated`, `thinking`.
- `vram_gb`: scheduling estimate = Hub safetensors size at the pinned revision + 3 GB, rounded up (Ministral: FP8 file dequantized to bf16, so 2×). Measured peaks land in each run's `complete.json` as `peak_gpu_gb`.
- Optional: `prefill` (text appended after the template prefill), `chat_kwargs` (passed to `apply_chat_template`), `fp8_dequantize`, `trust_remote_code` (default false), `blocked` (reason; `run_inference.py` refuses the model). No model is blocked now.

**How the list was built** (experiment log 2026-10-20): the supervisor asked for 15 prompted LLMs; the user set the ≤ 8B cap so models are comparable (this replaced Qwen3.5-9B with -4B and Ministral 3 8B with 3B). Criteria: instruct checkpoints, fit on a 40 GB A100, released ≥ 2 weeks before, family diversity. Every repo was checked on its Hub page; shas, gating and licences come from the HF API. gpt-oss is excluded (21B total). Backups considered: Qwen3.5-2B, tiny-aya-global, InternLM3-8B, Qwen3Guard-Gen-8B.

**Granite Guardian → Jamba2-3B.** Granite Guardian 3.3-8B is a guard model whose chat template judges its own "harm" criterion unless given a `guardian_config`; an adapter would have given it a prompt format no other model gets. It was blocked, never run, and replaced by Jamba2-3B (top pick of a candidate check: new family, plain ChatML template whose first reply token can be Yes/No, stock transformers, Apache-2.0). Runner-ups: LFM2.5-1.2B, Hunyuan-7B; rejected: InternLM3-8B and MiniCPM4.1-8B (> 8B, remote code), DeepSeek-R1-Distill-7B and Jamba-Reasoning-3B (forced reasoning), Mistral-7B v0.3 (family already present).

## 4.2 Task schemas (`src/schemas/`)

One pydantic `OutputSchema` subclass per task. Each sets four class variables (`dataset`, `task`, `task_type`, `column`) and the answer fields:

| Schema | Task | Type | Column | Fields |
|---|---|---|---|---|
| `CmsbBinary` | cmsb/binary | binary | `sexist` | `sexist: bool` |
| `EdosTaskA` | edos/task_a | binary | `sexist` | `sexist: bool` |
| `EdosTaskB` | edos/task_b | multiclass | `category` | `Literal[threats, derogation, animosity, prejudiced_discussions]` |
| `EdosTaskC` | edos/task_c | multiclass | `vector` | `Literal["1.1", …, "4.2"]` |
| `ExistTask11` | exist/task_1_1 | binary | `sexist` | `sexist: bool` |
| `ExistTask12` | exist/task_1_2 | multiclass | `intention` | `Literal[NO, REPORTED, JUDGEMENTAL, DIRECT]` |
| `ExistTask13` | exist/task_1_3 | multilabel | `labels` | 5 bools, aliased to the official names (`IDEOLOGICAL-INEQUALITY`, …) |

Schema helpers: `labels()` (class order: Literal args for multiclass, field aliases otherwise), `prompt_id()` (`{dataset}/{task}`), `json_schema()` (for constrained decoding; `extra="forbid"` rejects unknown keys), `from_prediction()` and `value()` (gold-file form).

Records (`src/schemas/records.py`):
- `Example(id, text, dataset, split)`: built by `Example.from_row(row, dataset)`; **`text` is `text_norm`**, not the raw text.
- `LabelScore(label, probs, mass, score, pred, top_k, refused=None)`: one per (post, label), or one per multiclass prompt.
- `Prediction(id, dataset, task, model_id, revision, backend, prompt_id, prompt_sha256, scores, raw_output, parsed, parse_error)`: one JSONL line.

### Three-place class-order invariant
The class order of a task is written in three places that must match:
1. `tasks` in `configs/data_prep.yaml`;
2. `src/schemas/tasks.py`;
3. `labels:` in `src/prompts/templates/{dataset}/{task}.yaml`.

`check_against_config(configs/data_prep.yaml)` (called at the start of every `run_inference.py` run) fails if a schema's type, column or class order differs from the config, or if a task exists in one but not the other. `PromptTemplate.load` fails if the template's labels differ from the schema. A new task therefore touches all three files (plus `01` for its columns).

## 4.3 Prompt templates (`src/prompts/`)

One YAML file per task under `src/prompts/templates/{dataset}/{task}.yaml`, all at `version: 1`:

| Key | Content |
|---|---|
| `source` | where the definitions come from |
| `system` | "You are an expert annotator who labels social media posts for sexism." (tweets for EXIST) |
| `yes_no` | binary/multilabel user message with `{name}`, `{definition}`, `{text}` |
| `options` | multiclass user message with `{options}` (lettered list) and `{text}` |
| `json` | generate-backend user message with `{fields}` and `{text}` |
| `prefill` | empty for every task (the grammar starts after it, so it must not contain JSON) |
| `labels` | label → `{name, definition}`, in schema order |

Definition sources:
- **CMSB:** adapted from Samory et al. 2021 (ICWSM), Table 2 and Sec. 2.2 (content or phrasing sexism).
- **EDOS A/B/C:** Kirk et al. 2023, SemEval-2023 Task 10 taxonomy. Some definitions were excerpted in our copy and are marked `TODO` in the YAML.
- **EXIST 1.1/1.2/1.3:** `EXIST2025_Annotation_guidelines_Tweets.pdf`, questions 1, 1–2 and 3 (examples left out).

Message shapes (the full rendered prompts for every task are generated into `results/final_eval/paper/prompts.tex`):
- **Yes/No** (binary, and each of the 5 EXIST 1.3 labels separately): `Definition of {name}: {definition}` / the post in triple quotes / `Is this post {name} according to the definition? Answer with Yes or No only.` (EXIST 1.3 asks "Does this tweet belong to the category {name}?").
- **Lettered options** (EDOS B, EDOS C, EXIST 1.2): `Which category … best describes this post?`, then `A) {name}: {definition}`, `B) …` in schema order, the post, and `Answer with the letter of one category only.` Letters A–D for 4 classes, A–K for EDOS C.

`render(tokenizer, messages, prefill, enable_thinking, chat)` applies the model's chat template with `add_generation_prompt=True` and `enable_thinking` (read by the Qwen, SmolLM3 and Nemotron templates, ignored by others), plus `chat.chat_kwargs`, then appends the template prefill and `chat.prefill`. The prompt ends exactly where the model's reply starts; for thinking models with thinking off it ends after an empty think block (checked on tokenizers).

Every prediction stores the template's `prompt_sha256` (SHA-256 of the YAML bytes). The full prompt text is not stored per label; it is rebuilt from the template sha plus the example id.

### Few-shot sets (`fewshot.py`, added 2026-10-23, no runs yet)
- **Format.** Shots are earlier chat turns. Each shot is a user turn (the same `yes_no` or `options` message, filled with the shot post) followed by an assistant turn (`Yes`/`No`, or the option letter). The real question comes last. The prompt still ends where the reply starts, so log-prob scoring is unchanged. With no shots, the messages and template sha256 are identical to zero-shot (checked on all 7 tasks).
- **Selection** (`select_shots`, seeded, in code):
  - Train only. Excluded: rows with `leak_group` or `dup_group`, `empty_norm` rows, rows without a hard label, and rows with more than 50 words. CMSB: annotated originals (`of_id` null). EXIST: `en`. EDOS B/C: gold-sexist rows.
  - Binary: k Yes + k No.
  - Multiclass: k per class.
  - EXIST 1.3: per label, k tweets with the label (Yes) + k sexist tweets without it (No), so the No shot marks the category boundary.
  - Order is a seeded shuffle.
- **Frozen files.** `configs/fewshot/{dataset}/{task}_k{k}_s{seed}.yaml` holds IDs and answers only (no post text in git), the filters, and the train parquet sha256. `freeze_shots` never overwrites a file. `load_shots` fills in `text_norm` and fails if the train file changed. Every model gets the same shots.
- **Frozen 2026-10-23 (k=1, seed 42):** all 7 tasks. Shots per prompt: binary 2; EDOS B 4, C 11; EXIST 1.2 4; EXIST 1.3 2 per label.
- **Measured prompt length** (Gemma-4-E4B tokenizer, first dev post, longest prompt per task; zero-shot → 1-shot): CMSB 168 → 503; EDOS A 95 → 305, B 149 → 773, C 358 → 4,300; EXIST 1.1 193 → 503, 1.2 232 → 973, 1.3 188 → 453. EDOS C repeats its 11-option list in every shot turn.
- Log-prob backend only; `--shots` with `generate` is refused.

## 4.4 Backends (`src/model/`)

### Loading (`loading.py`)
`load_hf_model(repo_id, revision, dtype=bfloat16, device_map="auto", token, fp8_dequantize, trust_remote_code)`:
- Tokenizer and model at the **pinned revision**, bf16.
- `fp8_dequantize` → `FineGrainedFP8Config(dequantize=True)` (Ministral 3).
- Tries `AutoModelForCausalLM`; on `ValueError` falls back to `AutoModelForImageTextToText` (multimodal checkpoints without a causal-LM mapping, e.g. `mistral3`).
- `model.eval()`.

### Log-prob backend (primary, `logprob.py`)
1. `next_token_probs`: tokenize the rendered prompt with `add_special_tokens=False` (the chat template already adds BOS), run one forward pass with `logits_to_keep=1`, take the last position's logits, cast to float32, softmax over the vocabulary. **Batch size 1**, one post per pass.
2. **Yes/No** (`score_yes_no`): P(Yes) = summed probability of the single-token variants of `Yes, " Yes", yes, " yes", YES`; P(No) likewise for `No, " No", no, " no", NO`. Variants that are not one token are dropped (all models had at least one; checked on tokenizers).
   - `mass = P(Yes) + P(No)`; `score = P(Yes)/mass` (None if mass is 0); `pred = score ≥ 0.5`.
3. **Options** (`score_options`): for each class, the summed probability of its letter and `" "+letter`; `mass` = sum over classes; `pred` = argmax class; `score` = its share of the mass.
4. `top_k` (default 10) next tokens with probabilities are logged so refusals can be inspected.
5. `score_example` returns one `LabelScore` per label (binary, multilabel) or one (multiclass).

`infer_single` wraps this into a `Prediction`:
- If `mass_threshold` is given, `refused = mass < threshold` per label; otherwise `refused` stays **None**.
- `parsed` (the schema dumped by alias) is set only if **every** label has a prediction and none is refused, because a refusal is its own outcome.

All 105 test runs used `mass_threshold = None`; the frozen cutoff τ = 0.5 is applied afterwards in `99_final_eval`.

### Generate backend (fallback, `generate.py`)
- `JsonGenerator(model, tokenizer, schema)` compiles the schema's JSON grammar once with outlines' `xgrammar` backend; generation is plain HF `generate`, greedy, `max_new_tokens` 256.
- Output: `raw_output` (always logged), `parsed` or a `parse_error` (usually truncation at `max_new_tokens`).
- **Cannot show refusals** (the grammar forces a schema-valid answer) and cannot restrict JSON whitespace (`whitespace_pattern` unsupported in xgrammar). Not used for any reported run.
- Thinking mode needs this backend, because log-prob reads the first reply token. But constrained decoding from the first token also suppresses thinking, so the planned Qwen3.5-4B think-on run needs a generate-then-score path first (not built).

## 4.5 `run_inference.py`

Runs one model over one processed split for one task, resumably.

```
python run_inference.py --dataset {cmsb,edos,exist} --task <key> [--split {dev,train,test}] --model <short name>
    [--model-id ID --model-name NAME --revision REV]   # override the registry for one-off runs
    [--backend {logprob,generate}] [--thinking] [--mass-threshold T]
    [--top-k 10] [--max-new-tokens 256] [--limit N] [--all-cmsb] [--seed 42]
    [--run-name EXISTING] [--log-every 25] [--shots K --shot-seed 42]
```
`--split` defaults to `dev`. Smoke test: `python run_inference.py --dataset cmsb --task binary --model gemma4-e4b --limit 5`.

Steps:
1. Refuse `--thinking` with `logprob`. Look up the registry entry (refuse blocked models); flags override it.
2. `check_against_config`; `get_schema(dataset, task)`.
3. `load_split(DATA_ROOT, dataset, split, schema, all_cmsb, limit)`:
   - **Test is label-blind:** only `id, text_norm, split` (plus CMSB `annotated`) are read.
   - CMSB: annotated only unless `--all-cmsb`.
   - EDOS B/C on train/dev: gold-sexist rows only. **On test, EDOS B/C run on all 4,000 posts** (the gold-sexist subset is chosen in `99`).
   - EXIST 1.2 and 1.3 run on all tweets (NO is a class / an empty set).
   - `--limit N`: the first N rows in file order.
4. Resolve the revision to a commit sha (`HfApi().model_info(...).sha`).
5. Run name: `{%Y%m%d-%H%M}_{dataset}_{task}_{model}_{method}_s{seed}` unless `--run-name` is given. `method = {backend}-{zs|fs{K}}[-think]-{split}[-n{limit}]`, e.g. `logprob-zs-test` or `logprob-fs1-test`.
   - With `--shots K`, the run loads `configs/fewshot/{dataset}/{task}_k{K}_s{shot_seed}.yaml` and refuses to start if any shot ID is in the evaluated split. It adds `shots`, `shot_seed`, `shots_file`, `shots_sha256` and `shot_ids` to `config.json`. These keys are absent in zero-shot runs, so unfinished zero-shot runs still resume.
6. `prepare_run_dir`: create the folder, or reopen an **unfinished** one only if every config key (except `command`) is unchanged. A finished run (`complete.json`, or `metrics.json` for older runs) is refused.
7. Write `config.json` and a "running" README; seed everything; load the model; write `environment.json`.
8. Skip ids already in `predictions.jsonl` (`read_done` drops a truncated last line left by a disconnect, and fails on a corrupt line elsewhere). For each remaining post, `infer_single`, write one JSON line and **flush**, so a disconnect loses at most the post in flight.
9. Check every id has a prediction (else exit with a resume hint).
10. Dev/train: `compute_metrics` → `metrics.json`. Test: no metrics.
11. Write `complete.json` (`n_predicted`, `finished`, `peak_gpu_gb`) and the final README.

## 4.6 Run folder contents

`OUTPUT_ROOT/{run_name}/`:

| File | Content |
|---|---|
| `config.json` | all CLI args, `model_id`, `model_name`, `revision_requested`, resolved `revision`, `chat_prefill`, `chat_kwargs`, `fp8_dequantize`, `trust_remote_code`, `method`, `data_file`, `data_sha256` (file bytes), `filters` (e.g. `label-blind, annotated only`), `n_examples`, `command` |
| `environment.json` | Python, platform, torch, transformers, CUDA, GPU name, git commit, git dirty flag, full `pip freeze` (rewritten on each resume) |
| `predictions.jsonl` | one `Prediction` per post |
| `metrics.json` | dev/train only |
| `complete.json` | marks a finished run |
| `README.md` | purpose, command, model@revision, seed, thinking, threshold, data hash, status, outcome |

Example `predictions.jsonl` line (log-prob, binary; values truncated):
```json
{"id": "...", "dataset": "cmsb", "task": "binary", "model_id": "google/gemma-4-E4B-it", "revision": "ee0ef602…",
 "backend": "logprob", "prompt_id": "cmsb/binary", "prompt_sha256": "2756fea9…",
 "scores": [{"label": "sexist", "probs": {"Yes": 0.998, "No": 0.0019}, "mass": 0.9999, "score": 0.998,
             "pred": true, "top_k": [["Yes", 0.998], ["No", 0.0019], …], "refused": null}],
 "raw_output": null, "parsed": {"sexist": true}, "parse_error": null}
```
For multiclass, `scores` has one entry with `label` = the target column and `probs` keyed by class; for EXIST 1.3 it has five entries, one per category.

## 4.7 Metrics on dev/train (`compute_metrics`)

**Strict:** a refused, zero-mass or unparsed answer is wrong. For binary/multilabel, a missing answer is replaced by the opposite of the gold; for multiclass by the sentinel `__unanswered__`, which matches no class.

| Type | Scored subset | Metrics |
|---|---|---|
| binary | all rows loaded | `macro_f1`, `f1_positive`, `accuracy`, `n_unanswered` |
| multiclass | non-null gold (EDOS: gold-sexist; EXIST 1.2: hard-majority tweets) | `macro_f1` over the classes, `accuracy` |
| multilabel | non-null `labels` (hard-majority tweets) | `macro_f1`, `micro_f1`, `per_label_f1`, `exact_match`, `n_unanswered_per_label`; official ICM is TODO |

Log-prob runs also store `mass` per label (`n, mean, min, p05, n_zero, refusal_rate` — the last only when a threshold is set). Generate runs store `parse_error_rate`.

The one dev result: `20261020_cmsb_binary_qwen3.5-4b_logprob-zs-dev_s42`, CMSB dev annotated-only (566 posts), macro-F1 0.802, F1-sexist 0.757, accuracy 0.813, 0 unanswered (`metrics.json` in that run folder, under `outputs/final_runs_result/outputs/`). It is a pipeline check, not part of the reported study.

## 4.8 Attribution helpers (no runs yet)

`src/utils/explain.py` (SHAP/LIME) works on saved models, not on `predictions.jsonl`: attributions need new forward passes on perturbed posts.
- `LabelScorer(model, tokenizer, template, label)`: posts → log P(Yes) − log P(No), scored exactly as `score_yes_no` (same prompt, one post per pass). Zero P(Yes) or P(No) gives NaN and is counted in `n_nan`, never filled.
- `shap_explain(score_fn, text, max_evals=300)`: SHAP Partition values per word on the log-odds scale; word split `\W+`, mask token `...`; only the post is masked, never the instructions.
- `lime_explain(score_fn, text, num_samples=500, num_features=10, seed=42)`: LIME weights for P(Yes), same word split, fixed `random_state`; raises if any perturbed post is unscorable.
- Yes/No tasks only (not EDOS B/C or EXIST 1.2). Cost is posts × labels × evaluations forward passes. Planned use: a small seeded **dev** sample, never test. Checked on a synthetic additive model only.

## 4.9 Run variants, sample mode and the text backend (added 2026-10-23, docs/12)

New `run_inference.py` flags. All default to the frozen setting, so v1 runs, their prompts, template sha256s and run names are unchanged (checked by `tests/test_prompts.py` and `tests/test_jobs.py`).

| Flag | Effect | Method-name tag |
|---|---|---|
| `--experiment-dir PATH` | Run folder goes to `{experiment}/runs/`; `{date}` = the experiment's creation date | — |
| `--sample N` | Sanity mode: N seeded random posts (label-blind), only in a sample experiment | `-sample-n{N}` |
| `--subset N` | A real, scored subset run of N seeded random posts | `sub{N}` |
| `--ids-file PATH` | Only these post ids; the config stores the file name and sha256, not the path | `ids-{stem}` |
| `--prompt-variant NAME` | Template `{task}.{NAME}.yaml` (v2, T1–T4, X), refused unless its sha256 matches `configs/prompt_variants.lock` | `{NAME}` |
| `--option-perm K` | Lettered options rotated left by K (multiclass only); probabilities stay keyed by class | `perm{K}` |
| `--backend text` | Greedy free-text reply (no grammar) read by the fixed parser in `src/model/text.py`; `parse_rule` is stored per prediction | backend `text` |

- **Prompt variants** (generated 2026-10-23, frozen in `configs/prompt_variants.lock`). In every variant the `labels:` block (definitions) is byte-identical to v1.
  - `v2` (EDOS B/C, EXIST 1.2): v1 plus a log-prob reply prefix `"Answer: "` (`logprob_prefill`).
  - `T1`/`T2`: reworded system and question sentences.
  - `T3`: post first.
  - `T4`: headed sections, no triple quotes.
  - `X` (binary tasks): asks for "Yes or No, then one sentence explaining why" (self-explanations).
- **Text parser** (docs/12 §3.3). For options, in order:
  1. a leading letter followed by `)`, `.`, `:`, `]` or a line end;
  2. "answer/option/category is X";
  3. exactly one whole-word class name;
  4. otherwise unparseable.

  For Yes/No, the first word.
- **New per-score fields** (absent in older runs): `tie` (the top two answers exactly equal; for options the earlier-shown class wins) and `rendered_sha256` (sha256 of the rendered prompt, which also catches text a chat template adds, such as SmolLM3's date).
- **Fix 2026-10-23:** `compute_metrics` now skips binary posts without a hard gold label (EXIST 1.1 dev has 45 such English tweets), as the multiclass and multilabel branches already did. Before the fix an EXIST 1.1 dev run crashed after inference. CMSB and EDOS results are unchanged (no missing gold).

