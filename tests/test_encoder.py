"""Encoder fine-tuning loop on CPU with a tiny random model built locally (no downloads)."""

import json
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"

import pandas as pd
import pytest

from src.model.encoder import EncoderHP, encoder_arrays, train_encoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (PROJECT_ROOT / "data" / "processed" / "dev").exists(), reason="needs the data"
)


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    from transformers import AutoTokenizer, RobertaConfig, RobertaModel

    from src.utils.registry import load_llm_registry

    spec = load_llm_registry(PROJECT_ROOT / "configs" / "llms.yaml")["exaone4-1.2b"]
    tok = AutoTokenizer.from_pretrained(spec.repo_id, revision=spec.revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = RobertaConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                        intermediate_size=64, max_position_embeddings=200, pad_token_id=tok.pad_token_id)  # fmt: skip
    path = tmp_path_factory.mktemp("tiny")
    RobertaModel(cfg).save_pretrained(path)  # headless, like the real base checkpoints
    tok.save_pretrained(path)
    return str(path)


@pytest.mark.parametrize(("label_col", "classes"), [("sexist", ["False", "True"]),
                                                    ("category", ["threats", "derogation", "animosity", "prejudiced_discussions"])])  # fmt: skip
def test_train_encoder_writes_label_blind_predictions(tmp_path, tiny_model, label_col, classes):
    dev = pd.read_parquet(PROJECT_ROOT / "data/processed/dev/edos_dev.parquet")
    dev = dev[dev[label_col].notna()] if label_col == "category" else dev
    train, dv, test = dev.iloc[:40], dev.iloc[40:60], dev.iloc[60:70][["id", "text_norm"]]
    hp = EncoderHP(epochs=2, batch_size=8, max_length=32, max_steps=6)
    out = train_encoder(tiny_model, None, train, dv, test, classes, label_col, hp, seed=13, run_dir=tmp_path)
    assert out["best_epoch"] in (1, 2) and len(out["history"]) >= 1
    recs = [json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text().splitlines()]
    assert len(recs) == 10 and set(recs[0]["probs"]) == set(classes)
    yp, probs = encoder_arrays(recs, [r["id"] for r in recs], classes)
    assert probs.shape == (10, len(classes)) and abs(probs.sum(1) - 1).max() < 1e-5
    assert (tmp_path / "dev_predictions.jsonl").exists() and (tmp_path / "dev_metrics.json").exists()
