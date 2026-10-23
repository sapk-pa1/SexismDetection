"""Secondary metrics of docs/12 §2.2-2.5: threshold-free, soft-label, label-free decision rules, risk-coverage.

All functions take arrays aligned with the strict arrays of `strict_arrays` (same ids, same order) and a shared
bootstrap index matrix, so every number is paired across models exactly like the frozen primary metric.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.schemas.base import OutputSchema

P_CLIP = 1e-12  # probability floor before taking logs, so log-odds stay finite (docs/12 §1)
CE_EPS = 1e-6  # cross-entropy floor (appendix only; docs/12 §2.3)


def label_scores(records: list[dict[str, Any]], schema: type[OutputSchema]) -> pd.DataFrame:
    """One row per post: log-odds `s_{label}`, `p_{label}` = P(Yes)/mass and `mass_{label}` per Yes/No label,
    or answer-forced class shares `q_{class}` plus `mass` for multiclass; `top_{label}` is the top-1 token."""
    rows = []
    for rec in records:
        row: dict[str, Any] = {"id": str(rec["id"])}
        for sc in rec["scores"]:
            top = sc["top_k"][0][0] if sc.get("top_k") else ""
            mass = sc["mass"]
            if schema.task_type == "multiclass":
                for c in schema.labels():
                    row[f"q_{c}"] = sc["probs"][c] / mass if mass > 0 else np.nan
                row |= {"mass": mass, "top": top, "tie": sc.get("tie")}
            else:
                lab = sc["label"]
                p_yes, p_no = sc["probs"]["Yes"], sc["probs"]["No"]
                row |= {
                    f"s_{lab}": np.log(max(p_yes, P_CLIP)) - np.log(max(p_no, P_CLIP)),
                    f"p_{lab}": p_yes / mass if mass > 0 else np.nan,
                    f"mass_{lab}": mass,
                    f"top_{lab}": top,
                }
        rows.append(row)
    return pd.DataFrame(rows).set_index("id")


def macro_f1_2d(y_rows: np.ndarray, yp_rows: np.ndarray, n_classes: int) -> np.ndarray:
    """Macro-F1 of each row pair (gold row, prediction row); codes >= n_classes are wrong (as macro_f1_counts)."""
    f1 = np.zeros((y_rows.shape[0], n_classes))
    for c in range(n_classes):
        t, p = y_rows == c, yp_rows == c
        tp, denom = (t & p).sum(1), t.sum(1) + p.sum(1)
        f1[:, c] = np.divide(2 * tp, denom, out=np.zeros(len(tp)), where=denom > 0)
    return f1.mean(1)


def _auroc_rows(y_rows: np.ndarray, s_rows: np.ndarray) -> np.ndarray:
    # Mann-Whitney AUROC per row; average ranks give ties 0.5, as in sklearn's roc_auc_score
    from scipy.stats import rankdata

    ranks = rankdata(s_rows, axis=1)
    n1 = y_rows.sum(1)
    n0 = y_rows.shape[1] - n1
    return ((ranks * y_rows).sum(1) - n1 * (n1 + 1) / 2) / np.maximum(n1 * n0, 1)


def _mcc_rows(y_rows: np.ndarray, yp_rows: np.ndarray) -> np.ndarray:
    tp = ((y_rows == 1) & (yp_rows == 1)).sum(1).astype(float)
    tn = ((y_rows == 0) & (yp_rows == 0)).sum(1).astype(float)
    fp = ((y_rows == 0) & (yp_rows == 1)).sum(1).astype(float)
    fn = ((y_rows == 1) & (yp_rows == 0)).sum(1).astype(float)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.divide(tp * tn - fp * fn, den, out=np.zeros(len(tp)), where=den > 0)


def binary_metrics(
    y: np.ndarray, s: np.ndarray, p: np.ndarray, yp: np.ndarray, idx: np.ndarray, chunk: int = 500
) -> dict[str, tuple[float, np.ndarray]]:
    """(point, bootstrap draws) of AUROC and AUPRC of the log-odds, Brier of P(Yes)/mass and MCC of the strict decision."""
    from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

    point = {
        "auroc": roc_auc_score(y, s),
        "auprc": average_precision_score(y, s),
        "brier": float(np.mean((p - y) ** 2)),
        "mcc": matthews_corrcoef(y, yp),
    }
    draws: dict[str, list[np.ndarray]] = {k: [] for k in point}
    for start in range(0, len(idx), chunk):
        block = idx[start : start + chunk]
        yr, sr = y[block], s[block]
        draws["auroc"].append(_auroc_rows(yr, sr))
        draws["auprc"].append(np.array([average_precision_score(a, b) for a, b in zip(yr, sr, strict=True)]))
        draws["brier"].append(((p[block] - yr) ** 2).mean(1))
        draws["mcc"].append(_mcc_rows(yr, yp[block]))
    return {k: (float(point[k]), np.concatenate(draws[k])) for k in point}


def soft_losses(q: np.ndarray, s: np.ndarray) -> dict[str, np.ndarray]:
    """Per-post L1, Brier and cross-entropy of predicted distributions q against annotator shares s.

    Binary inputs are 1-D shares of the positive class; multiclass inputs are (posts, classes) rows.
    L1 and Brier sum over classes for multiclass (binary L1 = |q - s|, Brier = (q - s)^2).
    """
    if q.ndim == 1:
        ce = -(s * np.log(np.maximum(q, CE_EPS)) + (1 - s) * np.log(np.maximum(1 - q, CE_EPS)))
        return {"l1": np.abs(q - s), "brier": (q - s) ** 2, "ce": ce}
    return {
        "l1": np.abs(q - s).sum(1),
        "brier": ((q - s) ** 2).sum(1),
        "ce": -(s * np.log(np.maximum(q, CE_EPS))).sum(1),
    }


def prior_matched(
    y: np.ndarray, s: np.ndarray, prior: float, idx: np.ndarray, tie_rank: np.ndarray
) -> tuple[float, np.ndarray, float]:
    """Macro-F1 when the top round(prior * n) posts by log-odds are labelled Yes (ties by post-id order).

    The rule is re-applied inside every resample. Returns (point, bootstrap draws, implied log-odds cut).
    """

    def decide(s_vec: np.ndarray, ties: np.ndarray) -> np.ndarray:
        k = int(round(prior * len(s_vec)))
        order = np.lexsort((ties, -s_vec))
        yp = np.zeros(len(s_vec), dtype=int)
        yp[order[:k]] = 1
        return yp

    yp = decide(s, tie_rank)
    point = float(macro_f1_2d(y[None, :], yp[None, :], 2)[0])
    cut = float(np.sort(s)[::-1][max(int(round(prior * len(s))) - 1, 0)])
    yp_rows = np.stack([decide(s[rows], tie_rank[rows]) for rows in idx])
    return point, macro_f1_2d(y[idx], yp_rows, 2), cut


def batch_calibrated(
    y: np.ndarray,
    q: np.ndarray,
    idx: np.ndarray,
    prior: np.ndarray | None = None,
    answered: np.ndarray | None = None,
    chunk: int = 200,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Batch calibration (Zhou et al. 2024): argmax_c [ln q_c - ln mean(q_c)] (+ ln prior_c if given).

    `q` is (posts, classes) answer-forced shares with class codes as in `y`; posts with `answered` False keep the
    strict non-answer code K (wrong). The batch mean is re-estimated inside every resample. Returns
    (point macro-F1, draws, point predictions).
    """
    n, k = q.shape
    logq = np.log(np.maximum(q, P_CLIP))
    bias = np.log(prior) if prior is not None else np.zeros(k)
    answered = np.ones(n, dtype=bool) if answered is None else answered

    def decide(lq: np.ndarray, mean_q: np.ndarray, ans: np.ndarray) -> np.ndarray:
        yp = np.argmax(lq - np.log(np.maximum(mean_q, P_CLIP))[..., None, :] + bias, axis=-1)
        return np.where(ans, yp, k)

    yp = decide(logq, q.mean(0), answered)
    point = float(macro_f1_2d(y[None, :], yp[None, :], k)[0])
    draws = []
    for start in range(0, len(idx), chunk):
        rows = idx[start : start + chunk]
        draws.append(macro_f1_2d(y[rows], decide(logq[rows], q[rows].mean(1), answered[rows]), k))
    return point, np.concatenate(draws), yp


def risk_coverage(correct: np.ndarray, confidence: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Selective risk at every coverage k/n (most confident first, stable order); returns (coverage, risk, AURC, risk@80%)."""
    order = np.argsort(-confidence, kind="mergesort")
    errors = np.cumsum(~correct[order])
    k = np.arange(1, len(correct) + 1)
    risk = errors / k
    cov = k / len(correct)
    at80 = float(risk[int(np.ceil(0.8 * len(correct))) - 1])
    return cov, risk, float(risk.mean()), at80


def per_class_f1(
    y: np.ndarray, yp: np.ndarray, n_classes: int, idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class F1 (point, bootstrap draws of shape (B, classes)); refusal codes >= n_classes count as wrong."""

    def f1(y_rows: np.ndarray, yp_rows: np.ndarray) -> np.ndarray:
        out = np.zeros((y_rows.shape[0], n_classes))
        for c in range(n_classes):
            t, p = y_rows == c, yp_rows == c
            tp, denom = (t & p).sum(1), t.sum(1) + p.sum(1)
            out[:, c] = np.divide(2 * tp, denom, out=np.zeros(len(tp)), where=denom > 0)
        return out

    return f1(y[None, :], yp[None, :])[0], f1(y[idx], yp[idx])


def fit_threshold(s: np.ndarray, y: np.ndarray) -> float:
    """Dev log-odds threshold maximising macro-F1 (docs/12 §3.1a): candidates are midpoints of sorted unique values
    plus 0; ties go to the smallest |t|. Decision: Yes iff s >= t."""
    u = np.unique(s)
    cands = np.unique(np.concatenate([(u[:-1] + u[1:]) / 2, [0.0]]))
    scores = macro_f1_2d(
        np.broadcast_to(y, (len(cands), len(y))), (s[None, :] >= cands[:, None]).astype(int), 2
    )
    best = scores.max()
    return float(min(cands[scores >= best - 1e-12], key=abs))


def fit_temperature(s: np.ndarray, y: np.ndarray) -> float:
    """Temperature T minimising the dev negative log-likelihood of sigmoid(s / T) (docs/12 §3.1c)."""
    from scipy.optimize import minimize_scalar
    from scipy.special import log_expit

    def nll(log_t: float) -> float:
        z = s / np.exp(log_t)
        return float(-(y * log_expit(z) + (1 - y) * log_expit(-z)).mean())

    return float(np.exp(minimize_scalar(nll, bounds=(-6, 6), method="bounded").x))


def fit_platt(s: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Platt scaling sigmoid(a*s + b) fitted on dev by maximum likelihood (docs/12 §3.1d)."""
    from scipy.optimize import minimize
    from scipy.special import log_expit

    def nll(w: np.ndarray) -> float:
        z = w[0] * s + w[1]
        return float(-(y * log_expit(z) + (1 - y) * log_expit(-z)).mean())

    a, b = minimize(nll, np.array([1.0, 0.0]), method="L-BFGS-B").x
    return float(a), float(b)


def pride_prior(letter_logp: np.ndarray) -> np.ndarray:
    """PriDe letter prior (Zheng et al., ICLR 2024) from (posts, cyclic perms, letters) log-probs of the letters.

    Averaging a post's log-probs over all cyclic orders cancels the content term, leaving log prior + const;
    the per-post priors are softmax-normalised and averaged over the estimation posts.
    """
    lp = letter_logp.mean(axis=1)
    lp = lp - lp.max(axis=1, keepdims=True)
    per_post = np.exp(lp) / np.exp(lp).sum(axis=1, keepdims=True)
    prior = per_post.mean(axis=0)
    return prior / prior.sum()


def pride_debias(letter_q: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Debiased class shares: observed letter shares divided by the letter prior, renormalised per post."""
    out = letter_q / prior[None, :]
    return out / out.sum(axis=1, keepdims=True)


def multilabel_macro_f1(y: np.ndarray, yp: np.ndarray, idx: np.ndarray) -> tuple[float, np.ndarray]:
    """Macro-F1 over labels (positive-class F1 per label, averaged) for (posts, labels) bool arrays; point and draws."""

    def f1(yr: np.ndarray, pr: np.ndarray) -> np.ndarray:
        tp = (yr & pr).sum(-2)
        denom = 2 * tp + (~yr & pr).sum(-2) + (yr & ~pr).sum(-2)
        return np.divide(2 * tp, denom, out=np.zeros(tp.shape), where=denom > 0).mean(-1)

    return float(f1(y, yp)), f1(y[idx], yp[idx])
