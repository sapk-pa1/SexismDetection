"""EvALL export: frozen rule unchanged, hierarchical 1.3 decoding (docs/12 §2.9), local PyEvALL scoring."""

import json
from pathlib import Path

import pytest

from src.schemas import get_schema
from src.utils.evall import evall_items, gold_subset, pyevall_scores

GOLD = Path("data/EXIST/evaluation/golds")
BASE = Path("data/EXIST/evaluation/baselines")
CATS = ["IDEOLOGICAL-INEQUALITY", "STEREOTYPING-DOMINANCE", "OBJECTIFICATION", "SEXUAL-VIOLENCE",
        "MISOGYNY-NON-SEXUAL-VIOLENCE"]  # fmt: skip


def rec13(i, yes_scores):
    return {
        "id": i,
        "scores": [
            {"label": c, "score": p, "pred": p >= 0.5, "refused": False}
            for c, p in zip(CATS, yes_scores, strict=True)
        ],
    }


def rec11(i, p, refused=False):
    return {"id": i, "scores": [{"label": "sexist", "score": p, "pred": p >= 0.5, "refused": refused}]}


def test_hierarchical_1_3_decoding():
    schema = get_schema("exist", "task_1_3")
    recs = [rec13("1", [0.9, 0.2, 0.1, 0.1, 0.1]), rec13("2", [0.9, 0.2, 0.1, 0.1, 0.1]),
            rec13("3", [0.4, 0.3, 0.1, 0.1, 0.2]), rec13("4", [0.9, 0.9, 0.9, 0.9, 0.9])]  # fmt: skip
    parents = [rec11("1", 0.8), rec11("2", 0.2), rec11("3", 0.7), rec11("4", 0.9, refused=True)]
    hard = {it["id"]: it["value"] for it in evall_items(recs, schema, False, parents)}
    assert hard == {
        "1": ["IDEOLOGICAL-INEQUALITY"],
        "2": ["NO"],
        "3": ["IDEOLOGICAL-INEQUALITY"],
    }  # 4 omitted
    soft = {it["id"]: it["value"] for it in evall_items(recs, schema, True, parents)}
    assert soft["2"]["NO"] == pytest.approx(0.8) and soft["1"]["IDEOLOGICAL-INEQUALITY"] == 0.9
    # Frozen rule (no parents): NO = prod(1 - P), categories decoded independently
    frozen = {it["id"]: it["value"] for it in evall_items(recs, schema, False)}
    assert frozen["2"] == ["IDEOLOGICAL-INEQUALITY"] and frozen["4"] == CATS


@pytest.mark.skipif(not GOLD.exists(), reason="needs the EXIST gold files")
def test_pyevall_reproduces_the_official_baseline_scores():
    import pandas as pd

    en = set(
        pd.read_parquet("data/processed/dev/exist_dev.parquet", columns=["id", "lang"])
        .query("lang == 'en'")
        .id.astype(str)
    )
    gold = gold_subset(GOLD / "EXIST2025_dev_task1_1_gold_hard.json", en)
    base = [
        b
        for b in json.loads((BASE / "EXIST2025_dev_task1_1_majority_class_hard.json").read_text())
        if b["id"] in en
    ]
    out = pyevall_scores(base, gold, ["ICM", "ICMNorm", "FMeasure"], "task_1_1")
    # Values from the 2026-10-23 PyEvALL check of the official majority-class baseline (docs/12 §3.2)
    assert out["ICM"] == pytest.approx(-0.4174, abs=1e-4) and out["ICM-Norm"] == pytest.approx(
        0.2889, abs=1e-4
    )
    assert out["F1"] == pytest.approx(0.3383, abs=1e-4)
    gold13 = gold_subset(GOLD / "EXIST2025_dev_task1_3_gold_hard.json", en)
    assert pyevall_scores(gold13, gold13, ["ICMNorm"], "task_1_3")["ICM-Norm"] == pytest.approx(1.0)
