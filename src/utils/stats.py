"""Uncertainty and agreement statistics over int-coded predictions: bootstrap CIs, McNemar, Holm, Fleiss' kappa, ECE."""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


def macro_f1_counts(y: np.ndarray, yp: np.ndarray, n_classes: int) -> float:
    """Macro-F1 over classes 0..n_classes-1; codes >= n_classes are wrong answers in no class (sklearn-equal)."""
    return float(_macro_f1_resampled(y, yp, n_classes, np.arange(len(y))[None, :])[0])


def _macro_f1_resampled(y: np.ndarray, yp: np.ndarray, n_classes: int, idx: np.ndarray) -> np.ndarray:
    # Per-class F1 = 2TP / (n_pred + n_true), 0 when both are 0 (sklearn zero_division=0)
    f1 = np.zeros((idx.shape[0], n_classes))
    for k in range(n_classes):
        t, p = (y == k)[idx], (yp == k)[idx]
        tp, denom = (t & p).sum(1), t.sum(1) + p.sum(1)
        f1[:, k] = np.divide(2 * tp, denom, out=np.zeros(len(tp)), where=denom > 0)
    return f1.mean(1)


def _accuracy_resampled(y: np.ndarray, yp: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return (y == yp)[idx].mean(1)


def bootstrap_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n) resample indices; reuse the same matrix across models so differences are paired."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def bootstrap_metric(
    y: np.ndarray, yp: np.ndarray, n_classes: int, idx: np.ndarray, metric: str = "macro_f1", chunk: int = 500
) -> np.ndarray:
    """Metric on every bootstrap resample (chunked to bound memory)."""
    parts = []
    for s in range(0, len(idx), chunk):
        block = idx[s : s + chunk]
        if metric == "macro_f1":
            parts.append(_macro_f1_resampled(y, yp, n_classes, block))
        elif metric == "accuracy":
            parts.append(_accuracy_resampled(y, yp, block))
        else:
            raise ValueError(f"unknown metric {metric}")
    return np.concatenate(parts)


def percentile_ci(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Two-sided percentile interval."""
    a = (1 - level) / 2
    return float(np.quantile(samples, a)), float(np.quantile(samples, 1 - a))


def paired_bootstrap_p(diff: np.ndarray) -> float:
    """Two-sided bootstrap p-value for H0: difference = 0, from paired resampled differences."""
    return float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean())))


def mcnemar_exact(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Exact McNemar test on per-post correctness of two systems; returns (a-only, b-only, p)."""
    ca, cb = a == y, b == y
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = 1.0 if n01 + n10 == 0 else binomtest(min(n01, n10), n01 + n10, 0.5).pvalue
    return n01, n10, float(p)


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in input order."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.maximum.accumulate((m - np.arange(m)) * p[order]).clip(max=1.0)
    out = np.empty(m)
    out[order] = adj
    return out.tolist()


def fleiss_kappa(codes: np.ndarray, n_categories: int) -> float:
    """Fleiss' kappa for an (items, raters) matrix of category codes 0..n_categories-1 (refusal may be a code)."""
    n_items, n_raters = codes.shape
    counts = np.stack([(codes == k).sum(1) for k in range(n_categories)], axis=1)
    p_i = ((counts * (counts - 1)).sum(1)) / (n_raters * (n_raters - 1))
    p_j = counts.sum(0) / (n_items * n_raters)
    p_bar, p_e = p_i.mean(), (p_j**2).sum()
    return float((p_bar - p_e) / (1 - p_e)) if p_e < 1 else float("nan")


def expected_calibration_error(
    y: np.ndarray, score: np.ndarray, n_bins: int = 10
) -> tuple[float, np.ndarray]:
    """ECE with equal-width bins on P(positive); returns (ece, rows of [bin_mid, mean_score, frac_pos, n])."""
    bins = np.clip((score * n_bins).astype(int), 0, n_bins - 1)
    rows, ece = [], 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            conf, acc = score[m].mean(), y[m].mean()
            ece += m.mean() * abs(conf - acc)
            rows.append([(b + 0.5) / n_bins, conf, acc, int(m.sum())])
    return float(ece), np.array(rows)


def simultaneous_ci(samples: np.ndarray, level: float = 0.95, m: int = 1) -> tuple[float, float]:
    """Bonferroni-simultaneous percentile interval for one of `m` comparisons (docs/12 §2.1)."""
    a = (1 - level) / (2 * m)
    return float(np.quantile(samples, a)), float(np.quantile(samples, 1 - a))


def format_p(p: float, n_boot: int) -> str:
    """p-value text; a bootstrap p of 0 is reported as below the resolution 1/B, never as 0."""
    if np.isnan(p):
        return ""
    return f"< {1 / n_boot:.0e}" if p == 0 else f"{p:.3g}"


def mde(diff: np.ndarray, alpha: float = 0.05, power: float = 0.8, m: int = 1) -> float:
    """Minimum detectable difference, normal approximation: (z_{1-alpha/2m} + z_power) * SD of the bootstrap diff."""
    from scipy.stats import norm

    return float((norm.ppf(1 - alpha / (2 * m)) + norm.ppf(power)) * np.std(diff, ddof=1))


def rank_stats(boots: dict[str, np.ndarray], level: float = 0.95) -> dict[str, dict[str, float]]:
    """P(best) and a rank interval per model from paired bootstrap scores (rank 1 = best, ties averaged)."""
    from scipy.stats import rankdata

    names = list(boots)
    mat = np.stack([boots[n] for n in names], axis=1)  # (B, models)
    ranks = rankdata(-mat, axis=1, method="average")
    top = mat == mat.max(axis=1, keepdims=True)
    p_best = (top / top.sum(axis=1, keepdims=True)).mean(axis=0)  # ties for first split equally
    a = (1 - level) / 2
    return {
        n: {"p_best": float(p_best[j]), "rank_median": float(np.median(ranks[:, j])),
            "rank_lo": float(np.quantile(ranks[:, j], a)), "rank_hi": float(np.quantile(ranks[:, j], 1 - a))}
        for j, n in enumerate(names)
    }  # fmt: skip


def spearman_ci(x: np.ndarray, y: np.ndarray, level: float = 0.95) -> tuple[float, float, float, float]:
    """Spearman rho, Fisher-z CI with SE sqrt(1.06/(n-3)) (Fieller et al. 1957), and its p-value."""
    from scipy.stats import norm, spearmanr

    rho, p = spearmanr(x, y)
    z, se = np.arctanh(rho), np.sqrt(1.06 / (len(x) - 3))
    q = norm.ppf(1 - (1 - level) / 2)
    return float(rho), float(np.tanh(z - q * se)), float(np.tanh(z + q * se)), float(p)


def _macro_f1_rows(y: np.ndarray, yp_rows: np.ndarray, n_classes: int) -> np.ndarray:
    # Macro-F1 of each row of predictions against one gold vector (same rule as macro_f1_counts)
    f1 = np.zeros((yp_rows.shape[0], n_classes))
    for k in range(n_classes):
        t, p = (y == k)[None, :], yp_rows == k
        tp, denom = (t & p).sum(1), t.sum() + p.sum(1)
        f1[:, k] = np.divide(2 * tp, denom, out=np.zeros(len(tp)), where=denom > 0)
    return f1.mean(1)


def approx_randomization_p(
    y: np.ndarray, a: np.ndarray, b: np.ndarray, n_classes: int, n_rounds: int = 10_000, seed: int = 42
) -> float:
    """Paired approximate-randomization p for a macro-F1 difference: swap the two systems per post with p = .5."""
    rng = np.random.default_rng(seed)
    observed = abs(macro_f1_counts(y, a, n_classes) - macro_f1_counts(y, b, n_classes))
    hits = 0
    for start in range(0, n_rounds, 500):
        swap = rng.random((min(500, n_rounds - start), len(y))) < 0.5
        d = _macro_f1_rows(y, np.where(swap, b, a), n_classes) - _macro_f1_rows(
            y, np.where(swap, a, b), n_classes
        )
        hits += int((np.abs(d) >= observed - 1e-12).sum())
    return (1 + hits) / (n_rounds + 1)


def kendalls_w(scores: np.ndarray) -> float:
    """Kendall's W of agreement between the rankings in the columns of an (items, raters) score matrix."""
    from scipy.stats import rankdata

    n, m = scores.shape
    ranks = rankdata(-scores, axis=0)
    r = ranks.sum(axis=1)
    return float(12 * ((r - r.mean()) ** 2).sum() / (m**2 * (n**3 - n)))


def variance_shares(mat: np.ndarray) -> dict[str, float]:
    """Shares of the total sum of squares of a (models, templates) score matrix: model, template, residual."""
    grand = mat.mean()
    ss_model = mat.shape[1] * ((mat.mean(axis=1) - grand) ** 2).sum()
    ss_tmpl = mat.shape[0] * ((mat.mean(axis=0) - grand) ** 2).sum()
    ss_total = ((mat - grand) ** 2).sum()
    resid = ss_total - ss_model - ss_tmpl
    return {"model": ss_model / ss_total, "template": ss_tmpl / ss_total, "residual": resid / ss_total}


def pigeonhole_bootstrap(
    y: np.ndarray, yp_by_template: np.ndarray, n_classes: int, n_boot: int = 2_000, seed: int = 42
) -> np.ndarray:
    """Prompt-averaged macro-F1 under a two-way bootstrap: posts and templates resampled independently (Owen 2007).

    Args:
        yp_by_template: (templates, posts) int-coded predictions, posts aligned with `y`.
    """
    rng = np.random.default_rng(seed)
    n_t, n = yp_by_template.shape
    out = np.empty(n_boot)
    for b in range(n_boot):
        posts = rng.integers(0, n, n)
        tmpls = rng.integers(0, n_t, n_t)
        out[b] = np.mean([macro_f1_counts(y[posts], yp_by_template[t, posts], n_classes) for t in tmpls])
    return out
