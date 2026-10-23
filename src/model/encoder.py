"""Fine-tuned encoder baselines (docs/12 §3.7): train on train, pick the epoch by dev macro-F1, predict test label-blind.

Plain PyTorch loop (no Trainer/accelerate). Hyperparameters are fixed in docs/12 and passed in by notebook 30.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EncoderHP:
    """Fixed hyperparameters (docs/12 §3.7); `epochs` depends on the task."""

    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup: float = 0.06
    batch_size: int = 16
    max_length: int = 128
    epochs: int = 4
    max_steps: int | None = None  # sample (sanity) mode only


def _batches(enc: dict[str, Any], labels: np.ndarray | None, order: np.ndarray, size: int):
    for start in range(0, len(order), size):
        rows = order[start : start + size]
        yield {k: v[rows] for k, v in enc.items()}, (labels[rows] if labels is not None else None)


def _predict(model: Any, enc: dict[str, Any], batch_size: int, device: str) -> np.ndarray:
    import torch

    model.eval()
    out = []
    with torch.inference_mode():
        for batch, _ in _batches(enc, None, np.arange(len(enc["input_ids"])), batch_size):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
            out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def train_encoder(
    model_id: str,
    revision: str,
    train: pd.DataFrame,
    dev: pd.DataFrame,
    test: pd.DataFrame,
    classes: list[str],
    label_col: str,
    hp: EncoderHP,
    seed: int,
    run_dir: Path,
    token: str | None = None,
) -> dict[str, Any]:
    """Fine-tune one encoder for one task and seed; writes predictions.jsonl (test) and dev_predictions.jsonl.

    `train`/`dev` carry `text_norm` and the label column; `test` carries only id and `text_norm` (label-blind).
    Binary tasks use classes ["False", "True"] with the label column cast to bool.
    """
    import torch
    from sklearn.metrics import f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    from src.utils.repro import set_seeds

    set_seeds(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    code = {c: k for k, c in enumerate(classes)}
    to_code = (lambda v: code[str(bool(v))]) if classes == ["False", "True"] else (lambda v: code[str(v)])
    y_train = np.array([to_code(v) for v in train[label_col]])
    y_dev = np.array([to_code(v) for v in dev[label_col]])

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, token=token)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, revision=revision, token=token, num_labels=len(classes)
    ).to(device)

    def encode(df: pd.DataFrame) -> dict[str, Any]:
        return dict(
            tok(
                list(df["text_norm"]),
                truncation=True,
                max_length=hp.max_length,
                padding=True,
                return_tensors="pt",
            )
        )

    enc_train, enc_dev, enc_test = encode(train), encode(dev), encode(test)
    steps_per_epoch = int(np.ceil(len(train) / hp.batch_size))
    total = hp.max_steps or steps_per_epoch * hp.epochs
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [
        {
            "params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)],
            "weight_decay": hp.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=hp.lr)
    sched = get_linear_schedule_with_warmup(opt, int(hp.warmup * total), total)
    rng = np.random.default_rng(seed)
    history, best_f1, best_state, step, start = [], -1.0, None, 0, time.time()
    for epoch in range(hp.epochs):
        model.train()
        for batch, labels in _batches(enc_train, y_train, rng.permutation(len(y_train)), hp.batch_size):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                out = model(
                    **{k: v.to(device) for k, v in batch.items()},
                    labels=torch.as_tensor(labels, device=device),
                )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if hp.max_steps and step >= hp.max_steps:
                break
        # Model selection on dev only (docs/12 §3.7); test is never scored here
        dev_f1 = f1_score(
            y_dev, _predict(model, enc_dev, 64, device).argmax(1), average="macro", zero_division=0
        )
        history.append({"epoch": epoch + 1, "steps": step, "dev_macro_f1": float(dev_f1)})
        if dev_f1 > best_f1:
            best_f1, best_state = (
                dev_f1,
                copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}),
            )
        if hp.max_steps and step >= hp.max_steps:
            break
    model.load_state_dict(best_state)
    for name, df, enc in (("dev_predictions.jsonl", dev, enc_dev), ("predictions.jsonl", test, enc_test)):
        probs = _predict(model, enc, 64, device)
        with (run_dir / name).open("w") as f:
            for i, p in zip(df["id"].astype(str), probs, strict=True):
                f.write(json.dumps({"id": i, "probs": dict(zip(classes, map(float, p), strict=True)),
                                    "pred": classes[int(p.argmax())]}) + "\n")  # fmt: skip
    best_epoch = max(history, key=lambda h: h["dev_macro_f1"])["epoch"]
    summary = {"history": history, "best_epoch": best_epoch, "best_dev_macro_f1": float(best_f1),
               "train_minutes": round((time.time() - start) / 60, 2), "hp": asdict(hp), "device": device}  # fmt: skip
    (run_dir / "dev_metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


def encoder_arrays(
    records: list[dict[str, Any]], ids: list[str], classes: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """(int predictions, class probabilities) aligned with `ids`, for scoring in 99 like the LLM strict arrays."""
    by_id = {r["id"]: r for r in records}
    probs = np.array([[by_id[i]["probs"][c] for c in classes] for i in ids])
    return probs.argmax(1), probs
