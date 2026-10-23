"""EXIST 2025 submission files in the official EvALL format, from log-prob prediction records."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

from src.schemas.base import OutputSchema

TEST_CASE = "EXIST2025"
EVALL_TASK = {"task_1_1": "task1_1", "task_1_2": "task1_2", "task_1_3": "task1_3"}
# Hierarchies passed to PyEvALL for the hierarchical tasks (EXIST 2025 overview; docs/12 §3.2)
HIERARCHY = {
    "task_1_2": {"YES": ["DIRECT", "REPORTED", "JUDGEMENTAL"], "NO": []},
    "task_1_3": {
        "YES": [
            "IDEOLOGICAL-INEQUALITY", "STEREOTYPING-DOMINANCE", "OBJECTIFICATION", "SEXUAL-VIOLENCE",
            "MISOGYNY-NON-SEXUAL-VIOLENCE",
        ],
        "NO": [],
    },
}  # fmt: skip


def _item(rec: dict[str, Any], schema: type[OutputSchema], soft: bool) -> Any | None:
    # None = refused or zero mass on some label; such posts are left out (strict: no answer)
    scores = rec["scores"]
    if any(s["refused"] or s["pred"] is None for s in scores):
        return None
    if schema.task_type == "binary":
        p = scores[0]["score"]
        return {"YES": p, "NO": 1 - p} if soft else ("YES" if scores[0]["pred"] else "NO")
    if schema.task_type == "multiclass":
        s = scores[0]
        return {c: v / s["mass"] for c, v in s["probs"].items()} if soft else s["pred"]
    yes = {s["label"]: s["score"] for s in scores}
    if soft:
        # NO share assumes independent labels: P(no category) = prod(1 - P(Yes))
        return yes | {"NO": math.prod(1 - p for p in yes.values())}
    active = [s["label"] for s in scores if s["pred"]]
    return active or ["NO"]


def _item_hierarchical(rec: dict[str, Any], parent: dict[str, Any] | None, soft: bool) -> Any | None:
    # EXIST 1.3 decoded under the model's own 1.1 answer (docs/12 §2.9); None = omitted (a refusal on either)
    if parent is None:
        return None
    p_score, scores = parent["scores"][0], rec["scores"]
    if (
        p_score["refused"]
        or p_score["pred"] is None
        or any(s["refused"] or s["pred"] is None for s in scores)
    ):
        return None
    yes = {s["label"]: s["score"] for s in scores}
    if soft:
        # The gold 1.3 NO share equals the 1.1 NO share, and category shares are unconditional
        return yes | {"NO": 1 - p_score["score"]}
    if not p_score["pred"]:
        return ["NO"]
    active = [s["label"] for s in scores if s["pred"]]
    return active or [max(yes, key=yes.get)]


def evall_items(
    records: list[dict[str, Any]],
    schema: type[OutputSchema],
    soft: bool,
    parent_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """[{test_case, id, value}] for answered posts; records must already carry the refusal flags.

    With `parent_records` (the same model's EXIST 1.1 records), task 1.3 is decoded hierarchically
    (docs/12 §2.9); without them the frozen 2026-10-20 export rule is used unchanged.
    """
    parents = {str(r["id"]): r for r in parent_records} if parent_records is not None else None
    out = []
    for rec in records:
        if parents is not None and schema.task_type == "multilabel":
            value = _item_hierarchical(rec, parents.get(str(rec["id"])), soft)
        else:
            value = _item(rec, schema, soft)
        if value is not None:
            out.append({"test_case": TEST_CASE, "id": str(rec["id"]), "value": value})
    return out


def gold_subset(gold_path: Path, ids: set[str]) -> list[dict[str, Any]]:
    """Official gold items restricted to the given ids (e.g. the English dev tweets)."""
    return [g for g in json.loads(gold_path.read_text()) if str(g["id"]) in ids]


def pyevall_scores(
    items: list[dict[str, Any]], gold: list[dict[str, Any]], metrics: list[str], task: str | None = None
) -> dict[str, float]:
    """Official EXIST metrics via PyEvALL (ICM, ICMNorm, FMeasure, ICMSoft, ICMSoftNorm, CrossEntropy).

    `task` selects the hierarchy for 1.2/1.3. Items missing from `items` are scored by PyEvALL as
    "no category" (lenient), which is why the omit variant is labelled as such.
    """
    from pyevall.evaluation import PyEvALLEvaluation
    from pyevall.utils.utils import PyEvALLUtils

    params = {
        PyEvALLUtils.PARAM_REPORT: PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED,
        PyEvALLUtils.PARAM_LOG_LEVEL: PyEvALLUtils.PARAM_OPTION_LOG_LEVEL_NONE,
    }
    if task in HIERARCHY:
        params[PyEvALLUtils.PARAM_HIERARCHY] = HIERARCHY[task]
    with tempfile.TemporaryDirectory() as tmp:
        pred_path, gold_path = Path(tmp) / "pred.json", Path(tmp) / "gold.json"
        pred_path.write_text(json.dumps(items))
        gold_path.write_text(json.dumps(gold))
        report = PyEvALLEvaluation().evaluate(str(pred_path), str(gold_path), metrics, **params)
    out = {}
    for m in report.report["metrics"].values():
        res = m["results"]
        out[m["acronym"]] = (
            float(res["average_per_test_case"]) if "average_per_test_case" in res else float("nan")
        )
    return out


def write_evall(
    items: list[dict[str, Any]], out_dir: Path, task: str, soft: bool, team: str, run: str
) -> Path:
    """Write one submission file named `{task}_{hard|soft}_{team}_{run}.json` (5 underscore parts, as the validator splits)."""
    if "_" in team or "_" in run:
        raise ValueError("team and run must not contain '_' (the validator splits the file name on it)")
    path = out_dir / f"{EVALL_TASK[task]}_{'soft' if soft else 'hard'}_{team}_{run}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=1))
    return path
