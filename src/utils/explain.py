"""Post-hoc word attributions (SHAP, LIME) for one label's Yes/No log-prob score."""

from __future__ import annotations

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
