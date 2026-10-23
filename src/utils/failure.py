"""Failure analysis of docs/12 §2.6-2.8: why a model did not say Yes (or the right class), plus lexicons and contrast sets.

Rules (codes, cut-offs, token classes, word lists) are frozen in docs/12 and configs/lexicons.yaml; nothing here
is tuned on results.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.schemas.base import OutputSchema

LEXICON_FILE = Path(__file__).resolve().parents[2] / "configs" / "lexicons.yaml"
CONFIDENT = 2.0  # |log-odds| at or above this is a confident answer (P(Yes) about 0.88)
CONFIDENT_SHARE = 0.8  # multiclass: share of answer mass on the predicted class for a confident wrong answer
NEAR_TIE = 0.05  # multiclass: gap between the top two answer-forced shares
LENGTH_BINS = [(0, 10, "<=10"), (11, 25, "11-25"), (26, 10**9, ">25")]


@lru_cache(maxsize=4)
def load_lexicons(path: Path = LEXICON_FILE) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def word_hits(texts: pd.Series, words: list[str]) -> pd.Series:
    """True where a text contains any of the words (lower-cased, whole words)."""
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(w.lower()) for w in words) + r")\b")
    return texts.fillna("").str.lower().str.contains(pattern).astype(bool)


def _swap_map(entries: dict[str, str]) -> dict[str, str]:
    # Forward entries first, then each value maps back to its first key unless it is itself a forward key
    mapping = {k.lower(): v.lower() for k, v in entries.items()}
    for k, v in entries.items():
        mapping.setdefault(v.lower(), k.lower())
    return mapping


def _match_case(src: str, out: str) -> str:
    if src.isupper() and len(src) > 1:
        return out.upper()
    return out.capitalize() if src[:1].isupper() else out


def gender_swap(text: str, entries: dict[str, str] | None = None) -> str:
    """Swap gendered words in one pass (both directions, case kept); the map is configs/lexicons.yaml gender_swap."""
    mapping = _swap_map(entries if entries is not None else load_lexicons()["gender_swap"])
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(w) for w in sorted(mapping, key=len, reverse=True)) + r")\b", re.I
    )
    return pattern.sub(lambda m: _match_case(m.group(0), mapping[m.group(0).lower()]), text)


def remove_words(text: str, words: list[str]) -> str:
    """Delete whole words (case-insensitive) and collapse the spaces left behind."""
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b", re.I)
    return re.sub(r"\s{2,}", " ", pattern.sub("", text)).strip()


def token_class(token: str, answer_tokens: set[str], classes: dict[str, list[str]] | None = None) -> str:
    """Class of a non-answer's first reply token: answer, markup, refusal, none_of_options, explanation or other."""
    classes = classes if classes is not None else load_lexicons()["non_answer_tokens"]
    tok = token.strip()
    if tok in answer_tokens:
        return "answer_token"
    if not re.search(r"[A-Za-z0-9]", tok):
        return "markup"
    for name in ("refusal", "none_of_options", "explanation"):
        if tok in classes[name]:
            return name
    return "other"


def _binary_reason(s: float, p: float, mass: float, threshold: float) -> str:
    if mass < threshold:
        return "non_answer"
    if p == 0.5:
        return "exact_tie"
    if s >= 0:
        return "confident_yes" if s >= CONFIDENT else "borderline_yes"
    return "confident_no" if s <= -CONFIDENT else "borderline_no"


def failure_table(
    scores: pd.DataFrame, gold: pd.Series, schema: type[OutputSchema], threshold: float = 0.5
) -> pd.DataFrame:
    """Outcome and reason code per (post, label) under the primary decision rule.

    Args:
        scores: `label_scores` output (index = post id).
        gold: per post id, the hard gold: bool (binary), class (multiclass) or list of labels (multilabel).
    """
    ids = [i for i in scores.index if i in gold.index and _has_gold(gold[i])]
    rows = []
    if schema.task_type == "multiclass":
        classes = schema.labels()
        letters = {chr(65 + j) for j in range(len(classes))}
        for i in ids:
            r = scores.loc[i]
            q = np.array([r[f"q_{c}"] for c in classes], dtype=float)
            answered = r["mass"] >= threshold
            pred = classes[int(np.nanargmax(q))] if np.isfinite(q).any() else None
            top2 = np.sort(q[np.isfinite(q)])[::-1][:2]
            gap = top2[0] - top2[1] if len(top2) == 2 else np.nan
            correct = answered and pred == gold[i]
            if not answered:
                reason = "non_answer"
            elif correct:
                reason = "correct"
            elif bool(r.get("tie")) or gap == 0:
                reason = "tie_break"
            elif gap < NEAR_TIE:
                reason = "near_tie"
            elif q.max() >= CONFIDENT_SHARE:
                reason = "confident_wrong"
            else:
                reason = "wrong"
            rows.append({
                "id": i, "label": schema.column, "gold": gold[i], "pred": pred if answered else None,
                "outcome": "non_answer" if not answered else ("correct" if correct else "wrong"),
                "reason": reason, "mass": r["mass"], "top_share": float(np.nanmax(q)) if np.isfinite(q).any() else np.nan,
                "top2_gap": gap, "top_token": r["top"],
                "token_class": token_class(str(r["top"]), letters) if not answered else "",
            })  # fmt: skip
        return pd.DataFrame(rows)
    labels = schema.labels()
    yes_no = {"Yes", "No", "yes", "no", "YES", "NO"}
    for i in ids:
        r = scores.loc[i]
        for lab in labels:
            g = bool(gold[i]) if schema.task_type == "binary" else lab in gold[i]
            s, p, mass = float(r[f"s_{lab}"]), float(r[f"p_{lab}"]), float(r[f"mass_{lab}"])
            answered = mass >= threshold
            yes = answered and p >= 0.5
            outcome = "non_answer" if not answered else {(True, True): "TP", (False, True): "FP",
                                                          (True, False): "FN", (False, False): "TN"}[(g, yes)]  # fmt: skip
            rows.append({
                "id": i, "label": lab, "gold": g, "pred": yes if answered else None, "outcome": outcome,
                "reason": _binary_reason(s, p, mass, threshold), "s": s, "p": p, "mass": mass,
                "top_token": r[f"top_{lab}"], "token_class": token_class(str(r[f"top_{lab}"]), yes_no) if not answered else "",
            })  # fmt: skip
    return pd.DataFrame(rows)


def _has_gold(v: Any) -> bool:
    if isinstance(v, (list, np.ndarray)):
        return True
    return v is not None and not (isinstance(v, float) and np.isnan(v))


def length_bin(n_words: float) -> str:
    """Length class of a post in words (docs/12 §2.6); a missing text counts as 0 words."""
    n = 0 if n_words is None or n_words != n_words else int(n_words)
    return next(name for lo, hi, name in LENGTH_BINS if lo <= n <= hi)


def mean_ci(values: np.ndarray, n_boot: int = 2_000, seed: int = 42) -> tuple[float, float, float]:
    """Mean with a 95% percentile bootstrap CI (posts resampled); NaN for an empty group."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    idx = np.random.default_rng(seed).integers(0, len(values), size=(n_boot, len(values)))
    draws = values[idx].mean(1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def rate_table(
    df: pd.DataFrame, by: list[str], flag: str, n_boot: int = 2_000, seed: int = 42
) -> pd.DataFrame:
    """Rate of a boolean column per group with bootstrap CIs and group sizes (descriptive, docs/12 §2.6)."""
    rows = []
    for key, grp in df.groupby(by, dropna=False, observed=True):
        rate, lo, hi = mean_ci(grp[flag].to_numpy(dtype=float), n_boot, seed)
        rows.append(dict(zip(by, key if isinstance(key, tuple) else (key,), strict=True)) | {
            "n": len(grp), f"{flag}_rate": rate, "ci_lo": lo, "ci_hi": hi})  # fmt: skip
    return pd.DataFrame(rows)


def contrast_pairs(meta: pd.DataFrame) -> pd.DataFrame:
    """(original, modification) id pairs among the scored posts, from CMSB `of_id` (docs/12 §2.8)."""
    scored = set(meta["id"])
    mods = meta[meta["of_id"].notna()]
    pairs = mods.assign(orig=lambda d: d["of_id"].astype(str))[["orig", "id"]].rename(columns={"id": "mod"})
    return pairs[pairs["orig"].isin(scored)].reset_index(drop=True)


def contrast_metrics(
    pairs: pd.DataFrame, gold: pd.Series, pred: pd.Series, s: pd.Series, n_boot: int = 2_000, seed: int = 42
) -> dict[str, Any]:
    """Pair accuracy and the within-pair log-odds gap s(original) - s(modification), with pair-bootstrap CIs."""
    valid = pairs[
        gold.loc[pairs["orig"]].to_numpy().astype(bool) & ~gold.loc[pairs["mod"]].to_numpy().astype(bool)
    ]
    both = (pred.loc[valid["orig"]].to_numpy() == 1) & (pred.loc[valid["mod"]].to_numpy() == 0)
    gap = s.loc[valid["orig"]].to_numpy() - s.loc[valid["mod"]].to_numpy()
    acc, acc_lo, acc_hi = mean_ci(both.astype(float), n_boot, seed)
    g, g_lo, g_hi = mean_ci(gap, n_boot, seed)
    return {
        "n_pairs": len(pairs), "n_valid_pairs": len(valid), "n_excluded_pairs": len(pairs) - len(valid),
        "pair_accuracy": acc, "pair_accuracy_lo": acc_lo, "pair_accuracy_hi": acc_hi,
        "logodds_gap": g, "logodds_gap_lo": g_lo, "logodds_gap_hi": g_hi,
    }  # fmt: skip
