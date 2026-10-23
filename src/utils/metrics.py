"""Strict metrics over predictions.jsonl records, plus Yes/No mass summaries for refusals."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.schemas.base import OutputSchema

UNANSWERED = "__unanswered__"  # multiclass stand-in for a refused/unparsed answer; always wrong


def answers(rec: dict[str, Any], schema: type[OutputSchema]) -> dict[str, Any]:
    """Per-label answer from one prediction; None marks refused, zero-mass or unparsed."""
    keys = [schema.column] if schema.task_type == "multiclass" else schema.labels()
    if rec["backend"] in ("generate", "text"):
        parsed = rec["parsed"] or {}
        return {k: parsed.get(k) for k in keys}
    return {s["label"]: None if s["refused"] else s["pred"] for s in rec["scores"]}


def mass_summary(records: list[dict[str, Any]], threshold: float | None) -> dict[str, Any]:
    """Yes/No (or option) mass per label; refusal rate only once a threshold is set."""
    by_label: dict[str, list[float]] = {}
    for rec in records:
        for s in rec["scores"]:
            by_label.setdefault(s["label"], []).append(s["mass"])
    out = {}
    for label, masses in by_label.items():
        m = np.array(masses)
        out[label] = {
            "n": len(m),
            "mean": float(m.mean()),
            "min": float(m.min()),
            "p05": float(np.quantile(m, 0.05)),
            "n_zero": int((m == 0).sum()),
            "refusal_rate": float((m < threshold).mean()) if threshold is not None else None,
        }
    return out


def _decision_rule(schema: type[OutputSchema], backend: str) -> str:
    if backend == "generate":
        return "JSON-constrained greedy generation"
    if backend == "text":
        return "greedy free text read by the fixed parser (src/model/text.py); unparseable = wrong"
    if schema.task_type == "multiclass":
        return "argmax over options"
    return "P(Yes)/mass >= 0.5, provisional until per-label thresholds are tuned on dev"


def compute_metrics(
    df: pd.DataFrame,
    records: list[dict[str, Any]],
    schema: type[OutputSchema],
    *,
    split: str,
    backend: str,
    mass_threshold: float | None,
) -> dict[str, Any]:
    """Strict metrics: a refused, zero-mass or unparsed answer is scored as wrong."""
    gold = df.set_index("id")[schema.column]
    preds = {r["id"]: answers(r, schema) for r in records}
    ids = [i for i in gold.index if i in preds]
    metrics: dict[str, Any] = {
        "dataset": schema.dataset,
        "split": split,
        "task": schema.task,
        "task_type": schema.task_type,
        "scoring": "strict (unanswered = wrong)",
        "decision_rule": _decision_rule(schema, backend),
        "mass_threshold": mass_threshold,
        "n_predicted": len(records),
    }

    if schema.task_type == "binary":
        label = schema.labels()[0]
        # EXIST 1.1 has tweets with no hard majority (null gold); they have no hard score (CMSB/EDOS have none)
        ids = [i for i in ids if pd.notna(gold[i])]
        y = np.array([bool(gold[i]) for i in ids])
        raw = [preds[i][label] for i in ids]
        yp = np.array([(not g) if p is None else bool(p) for p, g in zip(raw, y, strict=True)])
        metrics |= {
            "subset": "posts with a hard majority label" if schema.dataset == "exist" else "all scored posts",
            "n_scored": len(ids),
            "n_unanswered": sum(p is None for p in raw),
            "macro_f1": f1_score(y, yp, average="macro", zero_division=0),
            "f1_positive": f1_score(y, yp, zero_division=0),
            "accuracy": accuracy_score(y, yp),
        }
    elif schema.task_type == "multiclass":
        classes = schema.labels()
        # EDOS B/C rows are already gold-sexist only; EXIST 1.2 drops tweets with no hard majority
        ids = [i for i in ids if pd.notna(gold[i])]
        y = [str(gold[i]) for i in ids]
        raw = [preds[i][schema.column] for i in ids]
        yp = [UNANSWERED if p is None else p for p in raw]
        metrics |= {
            "subset": "gold-sexist posts only"
            if schema.dataset == "edos"
            else "posts with a hard majority label",
            "n_scored": len(ids),
            "n_unanswered": sum(p is None for p in raw),
            "macro_f1": f1_score(y, yp, labels=classes, average="macro", zero_division=0),
            "accuracy": accuracy_score(y, yp),
        }
    else:
        labels = schema.labels()
        # Tweets with no hard majority (null labels) have no hard gold; soft eval is separate
        ids = [i for i in ids if gold[i] is not None]
        y = np.array([[lab in gold[i] for lab in labels] for i in ids])
        raw = [[preds[i][lab] for lab in labels] for i in ids]
        yp = np.array(
            [
                [(not g) if p is None else bool(p) for p, g in zip(r, gr, strict=True)]
                for r, gr in zip(raw, y, strict=True)
            ]
        )
        metrics |= {
            "subset": "posts with a hard majority label",
            "n_scored": len(ids),
            "n_unanswered_per_label": {lab: sum(r[j] is None for r in raw) for j, lab in enumerate(labels)},
            "macro_f1": f1_score(y, yp, average="macro", zero_division=0),
            "micro_f1": f1_score(y, yp, average="micro", zero_division=0),
            "per_label_f1": dict(
                zip(labels, f1_score(y, yp, average=None, zero_division=0).tolist(), strict=True)
            ),
            "exact_match": float((y == yp).all(axis=1).mean()),
            "official_icm": "TODO: score with the official EXIST evaluation",
        }

    if backend == "logprob":
        metrics["mass"] = mass_summary(records, mass_threshold)
    else:
        metrics["parse_error_rate"] = float(np.mean([r["parse_error"] is not None for r in records]))
        metrics["refusal_note"] = (
            "unparseable replies count as wrong"
            if backend == "text"
            else "constrained decoding cannot show refusals"
        )
    return {k: float(v) if isinstance(v, np.floating) else v for k, v in metrics.items()}


def apply_mass_threshold(records: list[dict[str, Any]], threshold: float | None) -> list[dict[str, Any]]:
    """Copies with `refused = mass < threshold` per label; None returns the records unchanged."""
    if threshold is None:
        return records
    out = []
    for rec in records:
        scores = [s | {"refused": bool(s["mass"] < threshold)} for s in rec["scores"]]
        out.append(rec | {"scores": scores})
    return out


def refusal_rates(records: list[dict[str, Any]], threshold: float | None) -> dict[str, float]:
    """Share of posts refused per label (mass below threshold); 0 for every label when threshold is None."""
    return {lab: s["refusal_rate"] or 0.0 for lab, s in mass_summary(records, threshold).items()}


def answered_ids(records: list[dict[str, Any]], schema: type[OutputSchema]) -> set[str]:
    """IDs whose every label was answered (not refused, non-zero mass)."""
    return {r["id"] for r in records if all(v is not None for v in answers(r, schema).values())}


def strict_arrays(
    df: pd.DataFrame, records: list[dict[str, Any]], schema: type[OutputSchema]
) -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
    """Int-coded gold and strict predictions for binary/multiclass, on the posts `compute_metrics` scores.

    Returns (ids, y, yp, classes). Binary classes are ["False", "True"] and a refusal becomes the wrong
    class; multiclass refusals get code len(classes), which counts as wrong and belongs to no class.
    """
    if schema.task_type == "multilabel":
        raise ValueError("strict_arrays covers binary and multiclass tasks only")
    gold = df.set_index("id")[schema.column]
    preds = {r["id"]: answers(r, schema) for r in records}
    ids = [i for i in gold.index if i in preds and pd.notna(gold[i])]
    if schema.task_type == "binary":
        label = schema.labels()[0]
        y = np.array([int(bool(gold[i])) for i in ids])
        raw = [preds[i][label] for i in ids]
        yp = np.array([1 - g if p is None else int(bool(p)) for p, g in zip(raw, y, strict=True)])
        return ids, y, yp, ["False", "True"]
    classes = schema.labels()
    code = {c: k for k, c in enumerate(classes)}
    y = np.array([code[str(gold[i])] for i in ids])
    yp = np.array([len(classes) if (p := preds[i][schema.column]) is None else code[p] for i in ids])
    return ids, y, yp, classes
