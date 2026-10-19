# notebooks/

## Naming
`NN_short_name.ipynb`. The number gives the run order:

| Range | Purpose |
|---|---|
| 00 | Colab setup: mount Drive, paths, installs, secrets |
| 01 | Data prep: load datasets, build and check splits |
| 1x | Encoders: baselines, vanilla NeoBERT, HEFT |
| 2x | LLMs: log-probability baselines, refusal analysis |
| 3x | Explainability: LIME, SHAP, surrogate models |
| 4x | Statistics: bootstrap CIs, significance tests |
| 99 | `99_final_eval`: the only notebook that reads the test set |

## Cell order for every notebook
1. Mount Drive, then set `PROJECT_ROOT`, `DATA_ROOT`, `OUTPUT_ROOT`, `HF_HOME`.
2. Pinned `pip install`.
3. `CONFIG` dict (dataset, task, model, method, seed, run_name). This is the only cell to edit between runs. If the notebook has a `configs/*.yaml`, this cell loads it and checks its keys; edit the YAML instead (01 uses `configs/data_prep.yaml`).
4. Set seeds.
5. Experiment cells.
6. Last cell: save config and metrics to `OUTPUT_ROOT/{run_name}/`.

## Rules
- "Restart and run all" must work from top to bottom.
- Colab disconnects: write predictions incrementally and resume from the last saved ID.
- Copy data to `/content` at the start for speed. Write outputs to Drive.
- Strip cell outputs before committing (`nbstripout`). Outputs can contain offensive text and large tables.
- If the same helper gets copied into 3 or more notebooks, flag it for the later `src/` refactor instead of copying it again.