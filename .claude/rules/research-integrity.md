# Research integrity: careful claims, reproducibility, documentation

Everything in this repo is research that will be reported in a paper. Treat every number, claim and change as something a reviewer will check.

## Claims
- Only state what the evidence shows. Every claim about a result must point to the run (`run_name`) and metric file it comes from.
- Never invent, estimate or "fill in" numbers, citations, dataset statistics or model details. If something is unknown, write `TODO` or say so.
- Separate observation from interpretation: "macro-F1 is 0.62 vs 0.59" is an observation; "HEFT is better" needs seeds, variance and, where possible, a significance test.
- Don't call a difference an improvement without results over multiple seeds (report mean ± std) and the same training budget, data split and preprocessing for both systems.
- Name the dataset, split, task, label set and metric (macro/micro, threshold) next to every reported number.
- Report negative results, failed runs and refusals honestly; don't drop runs that look bad. If a run is excluded, record why.
- Use hedged wording ("suggests", "on this dataset") for anything beyond what was measured. No claims of generalisation to unseen datasets or languages without testing them.
- Verify facts about papers, models and datasets against the original source before writing them down; mark unverified ones as such.

## LLM refusals in multi-label tasks
- A refusal is its own outcome, not a label decision. Never fill refused labels at random, with all zeros, with the majority class or by copying the binary decision.
- Record refusals per (example, label), not just per example. A model may answer "misogyny" but refuse "sexual violence" for the same post.
- With Yes/No log-prob scoring, log the probability mass on the Yes/No tokens. Flag a label as a soft refusal when that mass falls below a threshold. Pick the threshold on dev, then freeze and document it.
- Report three numbers for each LLM:
  - refusal rate per label
  - metrics on all examples, with refused labels scored as wrong (strict)
  - metrics on the subset of examples that every compared model answered, which is the fair head-to-head
- Any retry policy (rephrased prompt, prefilled reply) applies to all examples and models in the same way, is set before seeing test data and is logged. Never retry only the refused examples by hand.

## Reproducibility
- A result that can't be reproduced from the repo plus `OUTPUT_ROOT/{run_name}/` doesn't count.
- Every run saves: full config, seed(s), model repo ID and revision, dataset version/split, package versions (`pip freeze`), GPU type, and the git commit or notebook version used.
- Set and log all seeds (Python, NumPy, PyTorch, CUDA; `transformers.set_seed`). Note any known nondeterminism.
- No manual, unlogged steps. Data filtering, preprocessing, label mapping and threshold choices live in code, not in someone's head.
- Never overwrite a finished run; use a new `run_name`.
- Don't change hyperparameters, prompts or thresholds after looking at test results.

## Documentation after every task
- When a task is done (experiment, notebook change, data step, analysis), document it before moving on.
- Append an entry to `docs/experiment-log.md` (create it if missing) with: date, what was done and why, run name(s), key results with their source files, issues or deviations from the plan, and next steps.
- Each run folder also gets a short `README.md`: purpose, command/notebook used, config summary and outcome.
- Notebooks start with a short markdown cell stating their purpose, inputs and outputs, and end with a summary of findings.
- Record decisions (e.g. why a threshold, prompt or label mapping was chosen) where they are made, with the evidence behind them.
- Keep entries short, factual and dated; update them if a result is later found to be wrong rather than deleting the history.
