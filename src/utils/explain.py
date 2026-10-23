"""Post-hoc word attributions (SHAP, LIME) for one label's Yes/No log-prob score."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np

# Same word split for SHAP and LIME so their attributions are comparable
WORD_SPLIT = r"\W+"


class LabelScorer:
    """Callable posts -> log P(Yes)/P(No) for one label, scored exactly as in `score_yes_no`.

    NaN marks a post with zero P(Yes) or P(No); `n_nan` counts them instead of filling a value.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        template: Any,
        label: str,
        enable_thinking: bool = False,
        chat: Any = None,
    ) -> None:
        # Imported here: src/utils must load before HF_HOME is set, so no torch at import time
        from src.model.logprob import NO_VARIANTS, YES_VARIANTS, answer_token_ids

        self.model, self.tokenizer, self.template, self.label = model, tokenizer, template, label
        self.enable_thinking, self.chat = enable_thinking, chat
        self.yes_ids = answer_token_ids(tokenizer, YES_VARIANTS)
        self.no_ids = answer_token_ids(tokenizer, NO_VARIANTS)
        self.n_calls = 0
        self.n_nan = 0

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        from src.model.logprob import next_token_probs
        from src.prompts.builder import render

        out = np.empty(len(texts))
        # One post per forward pass, as in the main runs, so scores match them exactly
        for i, text in enumerate(texts):
            messages = self.template.yes_no_messages(str(text), self.label)
            prompt = render(self.tokenizer, messages, enable_thinking=self.enable_thinking, chat=self.chat)
            probs = next_token_probs(self.model, self.tokenizer, prompt)
            p_yes, p_no = probs[self.yes_ids].sum().item(), probs[self.no_ids].sum().item()
            out[i] = np.log(p_yes) - np.log(p_no) if p_yes > 0 and p_no > 0 else np.nan
        self.n_calls += len(texts)
        self.n_nan += int(np.isnan(out).sum())
        return out


def shap_explain(score_fn: LabelScorer, text: str, max_evals: int = 300) -> dict[str, Any]:
    """SHAP Partition values per word on the log-odds scale; they sum to f(text) - base_value."""
    import shap

    # Masked words become "..."; the instructions around the post are never masked
    masker = shap.maskers.Text(WORD_SPLIT, mask_token="...", collapse_mask_token=True)
    explainer = shap.Explainer(score_fn, masker)
    calls, nans = score_fn.n_calls, score_fn.n_nan
    sv = explainer([text], max_evals=max_evals, silent=True)[0]
    return {
        "method": "shap_partition",
        "tokens": [str(t) for t in sv.data],
        "values": [float(v) for v in sv.values],
        "base_value": float(sv.base_values),
        "max_evals": max_evals,
        "n_evals": score_fn.n_calls - calls,
        "n_nan": score_fn.n_nan - nans,
    }


def lime_explain(
    score_fn: LabelScorer,
    text: str,
    num_samples: int = 500,
    num_features: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """LIME word weights for P(Yes); stochastic, so the seed is part of the result."""
    from lime.lime_text import LimeTextExplainer

    def proba(texts: list[str]) -> np.ndarray:
        log_odds = score_fn(texts)
        # LIME's ridge fit can't take missing scores; fail rather than fill them
        if np.isnan(log_odds).any():
            raise ValueError(
                f"{int(np.isnan(log_odds).sum())} perturbed posts had zero Yes or No probability"
            )
        p_yes = 1 / (1 + np.exp(-log_odds))
        return np.column_stack([1 - p_yes, p_yes])

    explainer = LimeTextExplainer(class_names=["No", "Yes"], split_expression=WORD_SPLIT, random_state=seed)
    calls = score_fn.n_calls
    exp = explainer.explain_instance(
        text, proba, labels=(1,), num_features=num_features, num_samples=num_samples
    )
    return {
        "method": "lime",
        "weights": [(str(w), float(v)) for w, v in exp.as_list(label=1)],
        "intercept": float(exp.intercept[1]),
        "local_r2": float(exp.score),
        "num_samples": num_samples,
        "seed": seed,
        "n_evals": score_fn.n_calls - calls,
    }


def _p_class(s: float, yes: bool) -> float:
    # Probability of the predicted class from log-odds
    p_yes = 1 / (1 + np.exp(-s))
    return float(p_yes if yes else 1 - p_yes)


def _erase(tokens: list[str], drop: set[int], mode: str) -> str:
    # SHAP text tokens carry their trailing separators; only the word part is masked or deleted
    sub = "..." if mode == "mask" else ""
    return "".join(re.sub(r"\w+", sub, t) if i in drop else t for i, t in enumerate(tokens)).strip()


def faithfulness(
    score_fn: LabelScorer,
    shap_out: dict[str, Any],
    ks: Sequence[int] = (1, 3, 5),
    n_random: int = 5,
    seed: int = 42,
    modes: Sequence[str] = ("mask", "delete"),
) -> dict[str, float]:
    """ERASER comprehensiveness and sufficiency (DeYoung et al. 2020) of the top-k SHAP words (docs/12 §3.8).

    On the probability of the predicted class: comp = p(x) - p(x without the words), suff = p(x) - p(only the
    words). Words are ranked by their SHAP value toward the predicted class. Random baselines use n_random
    random word sets of the same size; `mask` replaces words with "...", `delete` removes them (OOD check).
    """
    tokens = shap_out["tokens"]
    words = [i for i, t in enumerate(tokens) if re.search(r"\w", t)]
    s0 = float(score_fn(["".join(tokens).strip()])[0])
    yes = s0 >= 0
    sign = 1 if yes else -1
    ranked = sorted(words, key=lambda i: -sign * shap_out["values"][i])
    rng = np.random.default_rng(seed)
    out = {"s_orig": s0, "pred_yes": float(yes), "n_words": float(len(words))}
    p0 = _p_class(s0, yes)
    for mode in modes:
        for k in ks:
            if k > len(words):
                continue
            sets = [set(ranked[:k])] + [
                set(rng.choice(words, size=k, replace=False).tolist()) for _ in range(n_random)
            ]
            texts = [_erase(tokens, d, mode) for d in sets] + [
                _erase(tokens, set(words) - d, mode) for d in sets
            ]
            s = score_fn(texts)
            comp = [p0 - _p_class(v, yes) for v in s[: len(sets)]]
            suff = [p0 - _p_class(v, yes) for v in s[len(sets) :]]
            out |= {f"comp_{mode}@{k}": comp[0], f"suff_{mode}@{k}": suff[0],
                    f"rand_comp_{mode}@{k}": float(np.mean(comp[1:])), f"rand_suff_{mode}@{k}": float(np.mean(suff[1:]))}  # fmt: skip
    return out


def lexicon_class(token: str, lexicons: dict[str, Any]) -> str:
    """gender_term, gendered_slur, profanity or other, for a SHAP token (lower-cased word part)."""
    word = re.sub(r"\W+", "", token.lower())
    for name in ("gendered_slurs", "profanity", "gender_terms"):
        if word in {w.lower() for w in lexicons[name]}:
            return {"gendered_slurs": "gendered_slur", "gender_terms": "gender_term"}.get(name, name)
    return "other"


def code_rationales(
    model: Any, tokenizer: Any, rationales: Sequence[str], codes: dict[str, str], chat: Any = None
) -> list[dict[str, Any]]:
    """Code free-text rationales into a fixed taxonomy with lettered-option log-prob scoring (docs/12 §3.8)."""
    from src.model.logprob import answer_token_ids, next_token_probs
    from src.prompts.builder import OPTION_LETTERS, render

    letters = list(OPTION_LETTERS[: len(codes)])
    options = "\n".join(
        f"{ltr}) {name}: {desc}" for ltr, (name, desc) in zip(letters, codes.items(), strict=True)
    )
    out = []
    for text in rationales:
        messages = [
            {"role": "system", "content": "You code short explanations that a classifier gave for its decision."},
            {"role": "user", "content": f'Explanation: """{text}"""\n\nWhich code describes the main reason given?\n\n'
                                        f"{options}\n\nAnswer with the letter of one code only."},
        ]  # fmt: skip
        probs = next_token_probs(model, tokenizer, render(tokenizer, messages, chat=chat))
        p = {
            name: probs[answer_token_ids(tokenizer, (ltr, f" {ltr}"))].sum().item()
            for ltr, name in zip(letters, codes, strict=True)
        }
        mass = sum(p.values())
        out.append(
            {"code": max(p, key=p.get), "share": max(p.values()) / mass if mass else None, "mass": mass}
        )
    return out


def _word(token: str) -> str:
    return re.sub(r"\W+", "", token.lower())


def attribution_agreement(shap_out: dict[str, Any], lime_out: dict[str, Any]) -> dict[str, float]:
    """Spearman rho of SHAP vs LIME weights on the words LIME kept, and top-3 word overlap (docs/12 §3.8)."""
    from scipy.stats import spearmanr

    shap_w: dict[str, float] = {}
    for tok, val in zip(shap_out["tokens"], shap_out["values"], strict=True):
        if _word(tok):
            shap_w[_word(tok)] = shap_w.get(_word(tok), 0.0) + val
    lime_w = {_word(w): v for w, v in lime_out["weights"] if _word(w) in shap_w}
    rho = (
        spearmanr([shap_w[w] for w in lime_w], list(lime_w.values()))[0] if len(lime_w) >= 3 else float("nan")
    )
    top_s = {w for w, _ in sorted(shap_w.items(), key=lambda x: -abs(x[1]))[:3]}
    top_l = {w for w, _ in sorted(lime_w.items(), key=lambda x: -abs(x[1]))[:3]}
    return {"shap_lime_rho": float(rho), "shap_lime_top3_overlap": len(top_s & top_l) / 3}


def lime_stability(lime_outs: Sequence[dict[str, Any]]) -> float:
    """Mean pairwise Spearman rho of LIME word weights across seeds, on the words both runs kept."""
    from scipy.stats import spearmanr

    rhos = []
    for a in range(len(lime_outs)):
        for b in range(a + 1, len(lime_outs)):
            wa = {_word(w): v for w, v in lime_outs[a]["weights"]}
            wb = {_word(w): v for w, v in lime_outs[b]["weights"]}
            common = sorted(set(wa) & set(wb))
            if len(common) >= 3:
                rhos.append(spearmanr([wa[w] for w in common], [wb[w] for w in common])[0])
    return float(np.mean(rhos)) if rhos else float("nan")


def counterfactual_texts(text: str, lexicons: dict[str, Any]) -> dict[str, str]:
    """The two rewrites of docs/12 §3.8: gender swap, and gendered slurs plus profanity removed."""
    from src.utils.failure import gender_swap, remove_words

    return {
        "swap": gender_swap(text, lexicons["gender_swap"]),
        "noslur": remove_words(text, lexicons["gendered_slurs"] + lexicons["profanity"]),
    }
