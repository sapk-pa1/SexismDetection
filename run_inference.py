"""Run one prompted LLM over one processed split for one task; resumable, one JSONL line per post.

Example (Colab: set os.environ["HF_TOKEN"] from userdata first, then `!python run_inference.py ...`):
    python run_inference.py --dataset exist --task task_1_3 --model-id google/gemma-4-E4B-it \
        --model-name gemma4-e4b --revision main --limit 5
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

from src.schemas import Example, check_against_config, get_schema  # noqa: E402
from src.utils import (  # noqa: E402
    compute_metrics,
    environment_info,
    get_hf_token,
    load_predictions,
    load_split,
    prepare_run_dir,
    read_done,
    resolve_revision,
    set_seeds,
    sha256_file,
    write_readme,
)

DATA_CONFIG = PROJECT_ROOT / "configs" / "data_prep.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True, choices=["cmsb", "edos", "exist"])
    p.add_argument("--task", required=True, help="task key under `tasks` in configs/data_prep.yaml")
    p.add_argument(
        "--split", default="dev", choices=["train", "dev"], help="test is read only in 99_final_eval"
    )
    p.add_argument("--model-id", required=True, help="Hugging Face repo ID")
    p.add_argument("--model-name", required=True, help="short name for the run name, e.g. gemma4-e4b")
    p.add_argument("--revision", required=True, help="branch, tag or commit; resolved to a commit sha")
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

    check_against_config(DATA_CONFIG)
    schema = get_schema(args.dataset, args.task)
    df, data_file, filters = load_split(
        DATA_ROOT, args.dataset, args.split, schema, args.all_cmsb, args.limit
    )

    token = get_hf_token()
    revision = resolve_revision(args.model_id, args.revision, token)
    method = f"{args.backend}-zs{'-think' if args.thinking else ''}-{args.split}"
    method += f"-n{args.limit}" if args.limit is not None else ""
    run_name = args.run_name or (
        f"{datetime.now():%Y%m%d-%H%M}_{args.dataset}_{args.task}_{args.model_name}_{method}_s{args.seed}"
    )
    run_dir = OUTPUT_ROOT / run_name
    pred_path = run_dir / "predictions.jsonl"

    config = {
        "run_name": run_name,
        **{k: v for k, v in vars(args).items() if k not in ("run_name", "revision", "log_every")},
        "revision_requested": args.revision,
        "revision": revision,
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
    model, tokenizer = load_hf_model(args.model_id, revision, token=token)
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
                model_id=args.model_id,
                revision=revision,
                backend=args.backend,
                template=template,
                generator=generator,
                mass_threshold=args.mass_threshold,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                enable_thinking=args.thinking,
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
    metrics = compute_metrics(
        df, records, schema, split=args.split, backend=args.backend, mass_threshold=args.mass_threshold
    )
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_readme(run_dir, config, "finished", metrics)
    print(json.dumps({k: v for k, v in metrics.items() if k != "mass"}, indent=2))
    if torch.cuda.is_available():
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
