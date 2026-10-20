# Code structure: thin entry points, helpers in modules

Scripts (e.g. `run_inference.py`) and notebooks are entry points. They hold the arguments or `CONFIG` cell, the paths, the main loop and saving. Helper functions live in `src/` and are imported.

## When to move helpers into a module
Move them as soon as any of these holds:
- A script or notebook has more than 3 helper functions, or more than ~100 lines of helper code.
- The same helper is needed in 2 or more scripts or notebooks. Move it; don't copy it.
- The helper is research logic that must behave the same in every run: data filters, metrics, refusal handling, seeds, run-folder I/O.

Small one-off code under these limits can stay where it is used.

## Where helpers go
- Group by concern, one concern per module: `src/utils/data.py`, `repro.py`, `run_io.py`, `metrics.py`. Model code goes in `src/model/`, prompts in `src/prompts/`, schemas in `src/schemas/`.
- Add to an existing module before creating a new one. Create a new module only for a new concern.
- Re-export public helpers in the package `__init__.py` and import them from there: `from src.utils import load_split`.

## How to write the module
- No hard-coded paths or global state. Take `DATA_ROOT`, `OUTPUT_ROOT` and settings as arguments. Paths are still set once, at the top of the entry point.
- Take explicit arguments, not the argparse namespace or the whole `CONFIG` dict.
- Import torch/transformers inside functions in modules that load before `HF_HOME` is set.
- One-line docstring per public function (see `code-comments.md`).

## Moving code safely
- A move must not change behaviour. Run the same smoke test before and after, and check the outputs match.
- Log the move in `docs/experiment-log.md`: what moved where, and the check that was run.
- In notebooks, add `PROJECT_ROOT` to `sys.path`. After editing a module, restart the runtime or use `importlib.reload`.
