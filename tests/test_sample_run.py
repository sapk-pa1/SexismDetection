"""End-to-end sample mode of run_inference.py on CPU: a stub model with a cached tokenizer, no downloads."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ["HF_HUB_OFFLINE"] = "1"  # tokenizer from hf_cache only; must be set before huggingface_hub loads

import pytest
import torch

import run_inference
from src.utils.experiments import open_experiment, runs_dir, sanity_table
from src.utils.registry import load_llm_registry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL = "exaone4-1.2b"  # tokenizer and chat template are cached; weights are not needed
pytestmark = pytest.mark.skipif(
    not (PROJECT_ROOT / "data" / "processed" / "test").exists(), reason="needs the processed data"
)


class StubModel:
    """Random next-token logits in place of an LLM; enough to drive the whole run pipeline."""

    device = torch.device("cpu")

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.gen = torch.Generator().manual_seed(0)

    def __call__(self, input_ids=None, logits_to_keep=1, **_):
        return SimpleNamespace(logits=torch.randn(1, 1, self.vocab_size, generator=self.gen))

    def generate(self, input_ids=None, max_new_tokens=32, **_):
        # Fixed free-text reply for the text backend: an option letter, then a short reason
        return torch.cat([input_ids, self.reply_ids[:, :max_new_tokens]], dim=1)


@pytest.fixture
def patched(monkeypatch):
    from transformers import AutoTokenizer

    spec = load_llm_registry(PROJECT_ROOT / "configs" / "llms.yaml")[MODEL]
    tok = AutoTokenizer.from_pretrained(spec.repo_id, revision=spec.revision)
    stub = StubModel(len(tok))
    stub.reply_ids = torch.tensor(
        [tok.encode("B) derogation, because it insults women.", add_special_tokens=False)]
    )
    monkeypatch.setattr("src.model.load_hf_model", lambda *a, **k: (stub, tok))
    monkeypatch.setattr(run_inference, "resolve_revision", lambda repo, rev, token: rev)
    monkeypatch.setattr(run_inference, "get_hf_token", lambda: None)
    monkeypatch.setattr(run_inference, "environment_info", lambda root: {"stub": True})


def run(monkeypatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["run_inference.py", "--model", MODEL, *args])
    run_inference.main()


def make_exp(tmp_path: Path, sample: bool) -> Path:
    return open_experiment(
        tmp_path,
        "smoke",
        {"x": 1},
        purpose="test",
        protocol_ref="-",
        project_root=PROJECT_ROOT,
        sample=sample,
    )


@pytest.mark.parametrize(
    ("dataset", "task", "split"),
    [
        ("cmsb", "binary", "test"),
        ("edos", "task_b", "test"),
        ("exist", "task_1_3", "test"),
        ("cmsb", "binary", "dev"),
    ],
)
def test_sample_run_end_to_end(tmp_path, monkeypatch, patched, dataset, task, split):
    exp = make_exp(tmp_path, sample=True)
    run(
        monkeypatch,
        "--dataset",
        dataset,
        "--task",
        task,
        "--split",
        split,
        "--sample",
        "3",
        "--experiment-dir",
        str(exp),
    )
    (run_dir,) = list(runs_dir(exp).iterdir())
    assert run_dir.name.endswith("-sample-n3_s42")
    cfg = json.loads((run_dir / "config.json").read_text())
    assert cfg["sample"] == 3 and cfg["experiment"] == exp.name and "random sample of 3" in cfg["filters"]
    assert len((run_dir / "predictions.jsonl").read_text().splitlines()) == 3
    assert (run_dir / "metrics.json").exists() == (split == "dev")  # test stays label-blind
    assert sanity_table(exp, [run_dir.name]).ok.all()


def test_sample_draw_is_reproducible(tmp_path, monkeypatch, patched):
    ids = []
    for name in ("a", "b"):
        exp = open_experiment(
            tmp_path / name, "smoke", {}, purpose="", protocol_ref="", project_root=PROJECT_ROOT, sample=True
        )
        run(
            monkeypatch,
            "--dataset",
            "edos",
            "--task",
            "task_a",
            "--split",
            "test",
            "--sample",
            "4",
            "--experiment-dir",
            str(exp),
        )
        (run_dir,) = list(runs_dir(exp).iterdir())
        ids.append(
            [json.loads(line)["id"] for line in (run_dir / "predictions.jsonl").read_text().splitlines()]
        )
    assert ids[0] == ids[1]


def test_sample_and_full_experiments_never_mix(tmp_path, monkeypatch, patched):
    full = make_exp(tmp_path, sample=False)
    with pytest.raises(SystemExit, match="--sample must be given exactly"):
        run(
            monkeypatch,
            "--dataset",
            "cmsb",
            "--task",
            "binary",
            "--split",
            "test",
            "--sample",
            "3",
            "--experiment-dir",
            str(full),
        )
    sample = make_exp(tmp_path, sample=True)
    with pytest.raises(SystemExit, match="--sample must be given exactly"):
        run(
            monkeypatch,
            "--dataset",
            "cmsb",
            "--task",
            "binary",
            "--split",
            "test",
            "--experiment-dir",
            str(sample),
        )


def test_variant_flags_end_to_end(tmp_path, monkeypatch, patched):
    exp = make_exp(tmp_path, sample=False)
    base = ["--dataset", "edos", "--task", "task_b", "--split", "test", "--experiment-dir", str(exp)]
    run(monkeypatch, *base, "--subset", "4", "--prompt-variant", "v2", "--option-perm", "1")
    ids = tmp_path / "ids.txt"
    first = json.loads(next(iter(runs_dir(exp).glob("*/predictions.jsonl"))).open().readline())
    assert first["prompt_id"] == "edos/task_b.v2"
    assert list(first["scores"][0]["probs"])[0] == "derogation"  # rotated: derogation is letter A
    assert first["scores"][0]["rendered_sha256"] and first["scores"][0]["tie"] in (True, False)
    ids.write_text(first["id"] + "\n")
    run(monkeypatch, *base, "--ids-file", str(ids), "--backend", "text", "--max-new-tokens", "16")
    text_run = next(p for p in runs_dir(exp).iterdir() if "text" in p.name)
    assert "ids-ids" in text_run.name
    rec = json.loads((text_run / "predictions.jsonl").read_text())
    assert rec["parsed"] == {"category": "derogation"} and rec["parse_rule"] == "leading_letter"
    cfg = json.loads((text_run / "config.json").read_text())
    assert cfg["ids_file"] == "ids.txt" and len(cfg["ids_file_sha256"]) == 64
    with pytest.raises(SystemExit, match="multiclass tasks only"):
        run(monkeypatch, "--dataset", "cmsb", "--task", "binary", "--split", "test", "--experiment-dir", str(exp),
            "--option-perm", "1")  # fmt: skip
