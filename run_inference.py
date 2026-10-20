"""Run one prompted LLM over one processed split for one task; resumable, one JSONL line per post.

Example (Colab: set os.environ["HF_TOKEN"] from userdata first, then `!python run_inference.py ...`):
    python run_inference.py --dataset exist --task task_1_3 --model gemma4-e4b --limit 5
Models come from configs/llms.yaml; --model-id/--model-name/--revision override them for one-off runs.
--split test is label-blind: predictions only, scored in notebooks/99_final_eval.ipynb.
Resume an interrupted run by passing the same arguments plus --run-name <existing run>.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "data"))
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", PROJECT_ROOT / "outputs"))
HF_HOME = Path(os.environ.get("HF_HOME", PROJECT_ROOT / "hf_cache"))
os.environ["HF_HOME"] = str(HF_HOME)  # must be set before transformers is imported
sys.path.insert(0, str(PROJECT_ROOT))

from src.prompts import ChatOptions  # noqa: E402
from src.schemas import Example, check_against_config, get_schema  # noqa: E402
from src.utils import (  # noqa: E402
    compute_metrics,
    environment_info,
    get_hf_token,
    load_llm_registry,
    load_predictions,
    load_split,
    make_run_name,
    method_name,
    prepare_run_dir,
    read_done,
    resolve_revision,
    set_seeds,
    sha256_file,
    write_readme,
)

DATA_CONFIG = PROJECT_ROOT / "configs" / "data_prep.yaml"
LLM_CONFIG = PROJECT_ROOT / "configs" / "llms.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True, choices=["cmsb", "edos", "exist"])
    p.add_argument("--task", required=True, help="task key under `tasks` in configs/data_prep.yaml")
    p.add_argument(
        "--split", default="dev", choices=["train", "dev", "test"], help="test: label-blind, no metrics"
    )
    p.add_argument("--model", default=None, help="short name in configs/llms.yaml, e.g. gemma4-e4b")
    p.add_argument("--model-id", default=None, help="Hugging Face repo ID (overrides --model)")
    p.add_argument("--model-name", default=None, help="short name for the run name (overrides --model)")
    p.add_argument("--revision", default=None, help="branch, tag or commit (overrides --model)")
    p.add_argument("--backend", default="logprob", choices=["logprob", "generate"])
    p.add_argument("--thinking", action="store_true", help="Qwen3.5 only; needs --backend generate")
    p.add_argument(
        "--mass-threshold", type=float, default=None, help="soft-refusal cutoff, once frozen on dev"
    )
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--limit", type=int, default=None, help="first N examples in file order (smoke tests)")
    p.add_argument("--all-cmsb", action="store_true", help="keep unannotated CMSB 'other' posts")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default=None, help="existing run to resume; new name if omitted")
    p.add_argument("--log-every", type=int, default=25)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.thinking and args.backend == "logprob":
        # With thinking on, the first reply token is the thinking block, not the answer
        sys.exit("--thinking needs --backend generate (log-prob scoring reads the first reply token)")

    # Registry entry first, then explicit flags override it
    spec = load_llm_registry(LLM_CONFIG)[args.model] if args.model else None
    if spec and spec.blocked:
        sys.exit(f"{spec.name} is blocked in configs/llms.yaml: {spec.blocked}")
    model_id = args.model_id or (spec and spec.repo_id)
    model_name = args.model_name or (spec and spec.name)
    requested_revision = args.revision or (spec and spec.revision)
    if not (model_id and model_name and requested_revision):
        sys.exit("give --model, or all of --model-id, --model-name and --revision")
    chat = ChatOptions(prefill=spec.prefill, chat_kwargs=spec.chat_kwargs) if spec else ChatOptions()

    check_against_config(DATA_CONFIG)
    schema = get_schema(args.dataset, args.task)
    df, data_file, filters = load_split(
        DATA_ROOT, args.dataset, args.split, schema, args.all_cmsb, args.limit
    )

    token = get_hf_token()
    revision = resolve_revision(model_id, requested_revision, token)
    method = method_name(args.backend, args.thinking, args.split, args.limit)
    run_name = args.run_name or make_run_name(
        f"{datetime.now():%Y%m%d-%H%M}", args.dataset, args.task, model_name, method, args.seed
    )
    run_dir = OUTPUT_ROOT / run_name
    pred_path = run_dir / "predictions.jsonl"

    config = {
        "run_name": run_name,
        **{
            k: v
            for k, v in vars(args).items()
            if k not in ("run_name", "model_id", "model_name", "revision", "log_every")
        },
        "model_id": model_id,
        "model_name": model_name,
        "revision_requested": requested_revision,
        "revision": revision,
        # Per-model prompt/loading options from configs/llms.yaml; part of the prompt with the template sha
        "chat_prefill": chat.prefill,
        "chat_kwargs": chat.chat_kwargs,
        "fp8_dequantize": bool(spec and spec.fp8_dequantize),
        "trust_remote_code": bool(spec and spec.trust_remote_code),
        "method": method,
        "data_file": str(data_file.relative_to(DATA_ROOT)),
        "data_sha256": sha256_file(data_file),
        "filters": ", ".join(filters) or "none",
        "n_examples": len(df),
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
    }
    try:
        prepare_run_dir(run_dir, config)
    except (FileExistsError, ValueError) as e:
        sys.exit(str(e))
    write_readme(run_dir, config, "running", None)

    import torch

    from src.model import JsonGenerator, infer_single, load_hf_model
    from src.prompts import PromptTemplate

    set_seeds(args.seed)
    model, tokenizer = load_hf_model(
        model_id,
        revision,
        token=token,
        fp8_dequantize=config["fp8_dequantize"],
        trust_remote_code=config["trust_remote_code"],
    )
    (run_dir / "environment.json").write_text(json.dumps(environment_info(PROJECT_ROOT), indent=2))
    template = PromptTemplate.load(schema)
    generator = JsonGenerator(model, tokenizer, schema) if args.backend == "generate" else None

    done = read_done(pred_path)
    todo = [row for row in df.to_dict("records") if row["id"] not in done]
    print(f"{run_name}: {len(df)} examples, {len(done)} done, {len(todo)} to go")
    start = time.time()
    with pred_path.open("a") as f:
        for n, row in enumerate(todo, 1):
            pred = infer_single(
                model,
                tokenizer,
                Example.from_row(row, args.dataset),
                schema,
                model_id=model_id,
                revision=revision,
                backend=args.backend,
                template=template,
                generator=generator,
                mass_threshold=args.mass_threshold,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                enable_thinking=args.thinking,
                chat=chat,
            )
            # Flush per line so a disconnect loses at most the example in flight
            f.write(pred.model_dump_json() + "\n")
            f.flush()
            if n % args.log_every == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}  {(time.time() - start) / n:.2f} s/example")

    records = load_predictions(pred_path)
    missing = set(df["id"]) - {r["id"] for r in records}
    if missing:
        sys.exit(f"{len(missing)} examples have no prediction; rerun with --run-name {run_name}")
    # Test is label-blind: no metrics here; thresholds are frozen on dev and applied in 99_final_eval
    metrics = None
    if args.split != "test":
        metrics = compute_metrics(
            df, records, schema, split=args.split, backend=args.backend, mass_threshold=args.mass_threshold
        )
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps({k: v for k, v in metrics.items() if k != "mass"}, indent=2))
    complete = {"n_predicted": len(records), "finished": datetime.now().isoformat(timespec="seconds")}
    if torch.cuda.is_available():
        # Measured peak, to check the vram_gb estimates used for parallel scheduling
        complete["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    (run_dir / "complete.json").write_text(json.dumps(complete, indent=2))
    write_readme(run_dir, config, "finished", metrics)
    if torch.cuda.is_available():
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
