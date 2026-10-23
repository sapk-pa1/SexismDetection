"""Post-hoc metrics (docs/12 §2): each checked against sklearn/scipy or a hand-worked case."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score

from src.schemas import get_schema
from src.utils.failure import (
    contrast_metrics,
    contrast_pairs,
    failure_table,
    gender_swap,
    remove_words,
    token_class,
)
from src.utils.scoring_extra import (
    _mcc_rows,
    batch_calibrated,
    binary_metrics,
    label_scores,
    macro_f1_2d,
    prior_matched,
    risk_coverage,
    soft_losses,
)
from src.utils.stats import (
    approx_randomization_p,
    bootstrap_indices,
    format_p,
    kendalls_w,
    mde,
    percentile_ci,
    pigeonhole_bootstrap,
    rank_stats,
    simultaneous_ci,
    spearman_ci,
    variance_shares,
)

RNG = np.random.default_rng(0)


def binary_case(n=400):
    y = RNG.integers(0, 2, n)
    s = y * 1.5 + RNG.normal(0, 1.5, n)
    s[:20] = 0.0  # ties, like the bf16 score grid
    p = 1 / (1 + np.exp(-s))
    return y, s, p, (p >= 0.5).astype(int)


def test_binary_metrics_match_sklearn():
    y, s, p, yp = binary_case()
    idx = bootstrap_indices(len(y), 50, 1)
    out = binary_metrics(y, s, p, yp, idx)
    assert out["auroc"][0] == pytest.approx(roc_auc_score(y, s))
    assert out["auprc"][0] == pytest.approx(average_precision_score(y, s))
    assert out["mcc"][0] == pytest.approx(matthews_corrcoef(y, yp))
    for b in range(5):
        r = idx[b]
        assert out["auroc"][1][b] == pytest.approx(roc_auc_score(y[r], s[r]))
        assert out["auprc"][1][b] == pytest.approx(average_precision_score(y[r], s[r]))
        assert out["mcc"][1][b] == pytest.approx(matthews_corrcoef(y[r], yp[r]))
        assert out["brier"][1][b] == pytest.approx(np.mean((p[r] - y[r]) ** 2))
    assert _mcc_rows(y[None, :], np.zeros((1, len(y)), dtype=int))[0] == 0.0


def test_macro_f1_2d_matches_sklearn_with_refusal_code():
    y = RNG.integers(0, 4, (3, 200))
    yp = RNG.integers(0, 5, (3, 200))  # code 4 = refusal, wrong and in no class
    got = macro_f1_2d(y, yp, 4)
    for j in range(3):
        assert got[j] == pytest.approx(
            f1_score(y[j], yp[j], labels=[0, 1, 2, 3], average="macro", zero_division=0)
        )


def test_prior_matched_hits_the_prior_and_breaks_ties_by_id():
    y, s, _, _ = binary_case(101)
    tie_rank = np.arange(101)
    idx = bootstrap_indices(101, 20, 2)
    point, draws, cut = prior_matched(y, s, 0.3, idx, tie_rank)
    assert len(draws) == 20 and 0 <= point <= 1
    s_tied = np.zeros(10)
    _, _, cut = prior_matched(np.array([0, 1] * 5), s_tied, 0.3, bootstrap_indices(10, 2, 0), np.arange(10))
    assert cut == 0.0


def test_batch_calibration_removes_a_constant_bias():
    # Every post leans to class 1 by the same factor; dividing by the batch mean recovers the true argmax
    y = np.array([0, 0, 1, 1])
    q = np.array([[0.4, 0.6], [0.45, 0.55], [0.1, 0.9], [0.05, 0.95]])
    point, draws, yp = batch_calibrated(y, q, bootstrap_indices(4, 10, 0))
    assert yp.tolist() == [0, 0, 1, 1] and point == 1.0
    _, _, yp_na = batch_calibrated(
        y, q, bootstrap_indices(4, 3, 0), answered=np.array([True, False, True, True])
    )
    assert yp_na[1] == 2  # non-answer keeps the wrong code K


def test_soft_losses_and_risk_coverage():
    q, s = np.array([0.9, 0.2]), np.array([0.8, 0.0])
    out = soft_losses(q, s)
    assert out["l1"].tolist() == pytest.approx([0.1, 0.2]) and out["brier"].tolist() == pytest.approx(
        [0.01, 0.04]
    )
    qm, sm = np.array([[0.7, 0.3], [0.5, 0.5]]), np.array([[1.0, 0.0], [0.0, 1.0]])
    assert soft_losses(qm, sm)["l1"].tolist() == pytest.approx([0.6, 1.0])
    correct = np.array([True, True, False, True])
    cov, risk, aurc, at80 = risk_coverage(correct, np.array([0.9, 0.8, 0.7, 0.1]))
    assert risk.tolist() == pytest.approx([0, 0, 1 / 3, 1 / 4]) and aurc == pytest.approx((1 / 3 + 1 / 4) / 4)
    assert at80 == pytest.approx(1 / 4)


def test_stats_helpers():
    x = RNG.normal(size=1000)
    assert simultaneous_ci(x, m=1) == pytest.approx(percentile_ci(x))
    lo14, hi14 = simultaneous_ci(x, m=14)
    assert lo14 < percentile_ci(x)[0] and hi14 > percentile_ci(x)[1]
    assert format_p(0.0, 10_000) == "< 1e-04" and format_p(0.0312, 10_000) == "0.0312"
    assert mde(x) == pytest.approx(2.8016 * np.std(x, ddof=1), rel=1e-3)
    boots = {
        "a": RNG.normal(0.8, 0.01, 500),
        "b": RNG.normal(0.7, 0.01, 500),
        "c": RNG.normal(0.79, 0.01, 500),
    }
    rs = rank_stats(boots)
    assert sum(v["p_best"] for v in rs.values()) == pytest.approx(1.0) and rs["b"]["rank_median"] == 3
    a, b = RNG.normal(size=15), RNG.normal(size=15)
    rho, lo, hi, p = spearman_ci(a, b)
    assert rho == pytest.approx(spearmanr(a, b)[0]) and lo < rho < hi
    m = RNG.normal(size=(15, 5))
    assert kendalls_w(np.repeat(m[:, :1], 5, axis=1)) == pytest.approx(1.0)
    assert sum(variance_shares(m).values()) == pytest.approx(1.0)
    y = RNG.integers(0, 2, 300)
    assert approx_randomization_p(y, y, y, 2, n_rounds=200) == 1.0
    draws = pigeonhole_bootstrap(y, np.stack([y, 1 - y]), 2, n_boot=50)
    assert draws.shape == (50,) and 0 <= draws.min() <= draws.max() <= 1
    assert kendalltau([1, 2, 3], [1, 2, 3])[0] == 1.0


def test_gender_swap_and_word_removal():
    entries = {"woman": "man", "women": "men", "she": "he", "her": "his", "him": "her", "his": "her"}
    assert gender_swap("Women should know SHE is not a man.", entries) == "Men should know HE is not a woman."
    assert gender_swap("he gave her his book", entries) == "she gave his her book"
    assert remove_words("you stupid Bitch, go away", ["bitch", "stupid"]) == "you , go away"


def test_token_class_order():
    assert token_class(" The", {"Yes", "No"}) == "explanation"
    assert token_class("**", {"Yes", "No"}) == "markup"
    assert token_class("None", {"A", "B"}) == "none_of_options"
    assert token_class("I", {"Yes", "No"}) == "refusal"
    assert token_class("B", {"A", "B"}) == "answer_token"
    assert token_class("xyz", {"A"}) == "other"


def test_failure_table_binary_and_multiclass():
    rec = lambda i, py, pn, top="Yes": {"id": i, "scores": [{"label": "sexist", "probs": {"Yes": py, "No": pn},  # noqa: E731
                                        "mass": py + pn, "top_k": [[top, 0.5]]}]}  # fmt: skip
    recs = [rec("1", 0.9, 0.05), rec("2", 0.01, 0.9), rec("3", 0.3, 0.6), rec("4", 0.1, 0.1, "The")]
    schema = get_schema("cmsb", "binary")
    ft = failure_table(
        label_scores(recs, schema), pd.Series({"1": True, "2": True, "3": True, "4": False}), schema
    )
    got = ft.set_index("id")[["outcome", "reason", "token_class"]].to_dict("index")
    assert got["1"]["outcome"] == "TP" and got["1"]["reason"] == "confident_yes"
    assert got["2"]["outcome"] == "FN" and got["2"]["reason"] == "confident_no"
    assert got["3"]["outcome"] == "FN" and got["3"]["reason"] == "borderline_no"
    assert got["4"]["outcome"] == "non_answer" and got["4"]["token_class"] == "explanation"

    schema = get_schema("edos", "task_b")
    mrec = lambda i, probs, top="A": {"id": i, "scores": [{"label": "category", "probs": probs, "mass": sum(probs.values()),  # noqa: E731
                                       "top_k": [[top, 0.5]], "tie": False}]}  # fmt: skip
    base = {"threats": 0.0, "derogation": 0.0, "animosity": 0.0, "prejudiced_discussions": 0.0}
    recs = [mrec("a", base | {"derogation": 0.9, "animosity": 0.05}), mrec("b", base | {"threats": 0.45, "animosity": 0.44}),
            mrec("c", base | {"derogation": 0.01}, top="None")]  # fmt: skip
    gold = pd.Series({"a": "animosity", "b": "animosity", "c": "derogation"})
    got = failure_table(label_scores(recs, schema), gold, schema).set_index("id")
    assert got.loc["a", "reason"] == "confident_wrong" and got.loc["b", "reason"] == "near_tie"
    assert got.loc["c", "reason"] == "non_answer" and got.loc["c", "token_class"] == "none_of_options"


def test_contrast_set():
    meta = pd.DataFrame({"id": ["o1", "m1", "m2", "o2", "m3"], "of_id": [None, "o1", "o1", None, "o2"]})
    pairs = contrast_pairs(meta)
    assert pairs.values.tolist() == [["o1", "m1"], ["o1", "m2"], ["o2", "m3"]]
    gold = pd.Series({"o1": True, "m1": False, "m2": False, "o2": True, "m3": True})
    pred = pd.Series({"o1": 1, "m1": 0, "m2": 1, "o2": 1, "m3": 1})
    s = pd.Series({"o1": 3.0, "m1": -1.0, "m2": 2.0, "o2": 1.0, "m3": 1.0})
    out = contrast_metrics(pairs, gold, pred, s, n_boot=100)
    assert out["n_valid_pairs"] == 2 and out["n_excluded_pairs"] == 1
    assert out["pair_accuracy"] == pytest.approx(0.5) and out["logodds_gap"] == pytest.approx(2.5)


def test_dev_fitted_rules():
    from src.utils.scoring_extra import fit_platt, fit_temperature, fit_threshold

    s = np.array([-3.0, -1.0, 0.5, 1.0, 2.0, 4.0])
    y = np.array([0, 0, 0, 1, 1, 1])
    t = fit_threshold(s, y)
    assert t == 0.75 and ((s >= t).astype(int) == y).all()
    assert fit_threshold(np.array([1.0, 1.0]), np.array([1, 1])) == 0.0  # ties -> smallest |t|
    y2, s2 = binary_case(2000)[:2]
    temp = fit_temperature(s2, y2)
    assert 0.5 < temp < 3.0
    a, b = fit_platt(s2, y2)
    assert a > 0


def test_pride_recovers_a_letter_prior():
    from src.utils.scoring_extra import pride_debias, pride_prior

    k, n = 4, 300
    prior = np.array([0.55, 0.2, 0.15, 0.1])
    content = RNG.dirichlet(np.ones(k), n)  # per-post class preference
    obs = np.empty((n, k, k))
    for perm in range(k):
        shown = np.roll(content, -perm, axis=1)  # letter j shows class (j + perm) % k
        p = shown * prior[None, :]
        obs[:, perm, :] = p / p.sum(1, keepdims=True)
    est = pride_prior(np.log(obs))
    assert np.allclose(est, prior, atol=1e-6)
    deb = pride_debias(obs[:, 0, :], est)
    assert np.allclose(deb, content / content.sum(1, keepdims=True), atol=1e-6)


def test_faithfulness_on_a_known_scorer():
    from src.utils.explain import _erase, faithfulness, lexicon_class

    def score(texts):  # log-odds: "hate" is the only strong evidence for Yes
        return np.array([3.0 * ("hate" in t) - 1.0 + 0.5 * ("women" in t) for t in texts])

    shap_out = {
        "tokens": ["I ", "hate ", "these ", "women, ", "honestly"],
        "values": [0.0, 3.0, 0.0, 0.5, 0.0],
    }
    assert _erase(shap_out["tokens"], {1}, "mask") == "I ... these women, honestly"
    assert _erase(shap_out["tokens"], {1}, "delete") == "I  these women, honestly"
    out = faithfulness(score, shap_out, ks=(1, 2), n_random=20, seed=0)
    assert out["pred_yes"] == 1.0
    assert out["comp_mask@1"] > out["rand_comp_mask@1"] > 0
    assert out["suff_mask@1"] < out["rand_suff_mask@1"]  # keeping only "hate" keeps the decision
    lex = {"gender_terms": ["women"], "gendered_slurs": ["bitch"], "profanity": ["damn"]}
    assert [lexicon_class(t, lex) for t in ["Women, ", "bitch", "damn!", "cat"]] == [
        "gender_term", "gendered_slur", "profanity", "other"]  # fmt: skip


def test_multilabel_macro_f1_matches_sklearn():
    from src.utils.scoring_extra import multilabel_macro_f1

    y = RNG.random((300, 5)) < 0.3
    yp = RNG.random((300, 5)) < 0.3
    idx = bootstrap_indices(300, 5, 3)
    point, draws = multilabel_macro_f1(y, yp, idx)
    assert point == pytest.approx(f1_score(y, yp, average="macro", zero_division=0))
    assert draws[2] == pytest.approx(f1_score(y[idx[2]], yp[idx[2]], average="macro", zero_division=0))


def test_attribution_agreement_and_stability():
    from src.utils.explain import attribution_agreement, counterfactual_texts, lime_stability

    shap_out = {
        "tokens": ["I ", "hate ", "these ", "women, ", "honestly"],
        "values": [0.1, 3.0, -0.2, 0.5, 0.0],
    }
    lime = {"weights": [("hate", 0.9), ("women", 0.2), ("these", -0.1), ("I", 0.05)]}
    out = attribution_agreement(shap_out, lime)
    assert out["shap_lime_rho"] == pytest.approx(1.0) and out["shap_lime_top3_overlap"] == 1.0
    assert lime_stability(
        [lime, {"weights": [("hate", 0.8), ("women", 0.3), ("these", -0.2)]}]
    ) == pytest.approx(1.0)
    lex = {"gender_swap": {"women": "men"}, "gendered_slurs": ["bitch"], "profanity": ["damn"]}
    assert counterfactual_texts("damn women", lex) == {"swap": "damn men", "noslur": "women"}


def test_compute_metrics_skips_binary_posts_without_hard_gold():
    from src.utils.metrics import compute_metrics

    schema = get_schema("exist", "task_1_1")
    df = pd.DataFrame({"id": ["1", "2", "3"], "sexist": pd.array([1, 0, pd.NA], dtype="Int64")})
    recs = [{"id": i, "backend": "logprob", "scores": [{"label": "sexist", "pred": p, "refused": False, "mass": 1.0}]}
            for i, p in (("1", True), ("2", False), ("3", True))]  # fmt: skip
    out = compute_metrics(df, recs, schema, split="dev", backend="logprob", mass_threshold=None)
    assert (
        out["n_scored"] == 2
        and out["macro_f1"] == 1.0
        and out["subset"] == "posts with a hard majority label"
    )
