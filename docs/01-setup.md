# 1 · Setup: environment, paths, packages, secrets

## Where things run

- The repo is cloned into **Google Drive** and the notebooks run on **Google Colab**. The default clone path in every notebook is `/content/drive/MyDrive/Sexism-research` (marked `TODO: set to your Drive clone`).
- Every notebook also runs **locally** from `notebooks/` with the project venv `.venv/` (Python 3.12). Locally, `PROJECT_ROOT = Path.cwd().parent`.
- GPU work (LLM inference) was done on a Colab **A100-SXM4-80GB** (`environment.json` of the test runs). Data prep, data analysis and the final evaluation run on CPU.

## Paths

Each notebook's first code cell sets four paths, once, and nothing else in the repo hard-codes a path:

| Variable | Default | Notes |
|---|---|---|
| `PROJECT_ROOT` | Drive clone (Colab) or the repo root (local) | |
| `DATA_ROOT` | `PROJECT_ROOT/data` | Raw and processed datasets; gitignored |
| `OUTPUT_ROOT` | `PROJECT_ROOT/outputs` | One folder per run; gitignored. On Colab it stays on Drive so runs survive disconnects |
| `HF_HOME` | `PROJECT_ROOT/hf_cache` | Hugging Face cache. `20_llm_runs` and `99_final_eval` use `/content/hf_cache` on Colab, because Drive is slow and too small for 15 models (~150 GB) |

`99_final_eval` adds `RUNS_ROOT = OUTPUT_ROOT/final_runs_result/outputs` (where the frozen 105-run batch was unpacked) and `RESULTS_ROOT = PROJECT_ROOT/results`.

`run_inference.py` reads the same paths from environment variables `DATA_ROOT`, `OUTPUT_ROOT`, `HF_HOME` (defaults: the repo folders). `20_llm_runs` passes them to each job process.

**`HF_HOME` must be set before `transformers` is imported**, otherwise the cache goes to the default location. For this reason `src/utils/` imports torch and transformers inside functions only, and `run_inference.py` sets `os.environ["HF_HOME"]` before importing anything from `src`.

**Speed on Colab.** Notebooks copy inputs from Drive to local disk before reading them: `01` copies the raw files to `/content/data`; `02` and `20` copy `data/processed` to `/content/data/processed`. Outputs are always written back to Drive.

## Packages

Pins live in two places that must agree: `requirements.txt` and the install cell of `notebooks/00_colab_setup.ipynb`. Each later notebook has a `PINS` dict, installs it on Colab, and asserts that the installed versions equal it (`installed {...} differ from PINS: install the pins and restart the kernel`).

| Package | Version | Used by |
|---|---|---|
| pandas | 3.0.6 | all |
| numpy | 2.5.4 | all |
| scikit-learn | 1.9.1 | metrics, 01 checks |
| pyarrow | 26.0.0 | Parquet I/O |
| pyyaml | 6.0.3 | configs |
| transformers | 5.20.1 | LLM loading and scoring |
| pydantic | 2.13.5 | schemas and records |
| outlines | 1.3.3 | grammar compilation (generate backend) |
| xgrammar | 0.2.8 | JSON grammar backend |
| shap | 0.54.1 | `src/utils/explain.py` |
| lime | 0.2.0.1 | `src/utils/explain.py` |
| matplotlib | 3.11.2 | pinned in `02` only |

Not pinned:
- **torch**: Colab's preinstalled build; its version is logged in each run's `environment.json` (the test batch used `2.11.0+cu130`, CUDA 13.0).
- **huggingface_hub**: comes with transformers; used by `resolve_revision`.
- **scipy, matplotlib, jsonschema** in `99_final_eval`: logged, not pinned (open item in [09](09-known-issues.md)).
- **mamba-ssm / causal-conv1d**: not installed, so Jamba2 and Falcon-H1 use transformers' PyTorch Mamba path (slower; can differ slightly in numbers from the CUDA kernels).

Notes:
- `pyproject.toml` says `requires-python = ">=3.12,<3.13"`, and its comment about `requirements.lock` is stale (there is no lock file).
- The Colab runtime used for the test batch reported **Python 3.13.15** (`environment.json`), outside that range. See [09](09-known-issues.md).
- If pandas was imported before pyarrow was installed, `read_parquet` breaks; `01` smoke-tests this and stops with "restart the kernel". After any `%pip install` that changed something, restart the runtime and run all again.

Lint and format (config in `pyproject.toml`, line length 110, rules `E F I B UP SIM`):
```
ruff check src run_inference.py
ruff format src run_inference.py
```
There are no tests yet; `pyproject.toml` already sets `testpaths = ["tests"]` and the markers `gpu`, `slow`, `network`.

## Secrets

- `HF_TOKEN` comes from **Colab secrets** (`google.colab.userdata.get("HF_TOKEN")`) or from the environment locally. `src.utils.get_hf_token` tries the env first, then Colab secrets.
- Never paste a token into a cell, a config or git. `20_llm_runs` passes the token to job processes through the environment only.
- Gated models: **Llama 3.1 8B** (manual approval) and **Aya Expanse 8B** (automatic approval). The token's account must have accepted both licences, otherwise the jobs fail with `403 ... Cannot access gated repo` (this happened to Aya on the first batch session). Gemma 4 E4B is **not** gated (the message printed by `00` saying so is outdated).

## `00_colab_setup.ipynb`

One-time checks, no outputs written:
1. Mount Drive and set the four paths.
2. Install the pins.
3. Assert that the expected raw files exist and print their sizes (CMSB data and annotations, EDOS aggregated and individual, EXIST train/dev JSON, EXIST dev 1.3 golds).
4. Create `OUTPUT_ROOT` and `HF_HOME`; print the Python version.
5. Read `HF_TOKEN` and print the HF user name (`whoami`).
6. Print the GPU name and memory, or "no GPU".

## Typical session order

1. `00_colab_setup` once per runtime.
2. `01_data_prep` (CPU) whenever raw data or the pipeline changes; otherwise it only verifies.
3. `02_data_analysis` (CPU).
4. `20_llm_runs` on a GPU runtime (or `run_inference.py` directly for one job).
5. `99_final_eval` (CPU), once, after the evaluation protocol is frozen.

Notebook conventions (cell order, "restart and run all" must work, strip outputs before committing with `nbstripout`) are in [`../notebooks/CLAUDE.md`](../notebooks/CLAUDE.md).
