"""Experiment folders: resume vs duplicate, manifest rebuild, and the guards 99 relies on."""

import json
from pathlib import Path

import pytest

from src.utils.experiments import (
    find_experiment,
    load_experiment_runs,
    open_experiment,
    read_experiment,
    register_experiment,
    runs_dir,
    sanity_table,
    write_manifest,
)
from src.utils.repro import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = {"tasks": [("cmsb", "binary")], "models": ["m1"], "max_parallel": 4}


def fake_run(
    parent: Path, name: str, split: str = "test", done: bool = True, sample: int | None = None
) -> Path:
    run = parent / name
    run.mkdir(parents=True)
    cfg = {"run_name": name, "dataset": "cmsb", "task": "binary", "split": split, "model": "m1",
           "model_name": "m1", "backend": "logprob", "seed": 42, "n_examples": 2, "sample": sample}  # fmt: skip
    (run / "config.json").write_text(json.dumps(cfg))
    (run / "predictions.jsonl").write_text('{"id": "1"}\n{"id": "2"}\n')
    if done:
        (run / "complete.json").write_text(json.dumps({"n_predicted": 2, "finished": "2026-10-24T10:00:00"}))
    return run


def make_exp(
    tmp_path: Path, name: str = "devsweep", sample: bool = False, config: dict | None = None
) -> Path:
    return open_experiment(
        tmp_path,
        name,
        config or CONFIG,
        purpose="test",
        protocol_ref="docs/12",
        project_root=PROJECT_ROOT,
        sample=sample,
    )


def test_open_twice_resumes_without_duplicating(tmp_path):
    first = make_exp(tmp_path)
    second = make_exp(tmp_path)
    assert first == second
    assert len(list((tmp_path / "experiments").iterdir())) == 1
    assert read_experiment(first)["config"]["tasks"] == [["cmsb", "binary"]]
    assert find_experiment(tmp_path, "devsweep") == first


def test_changed_config_is_refused_but_ignored_keys_may_differ(tmp_path):
    make_exp(tmp_path)
    with pytest.raises(ValueError, match="different settings"):
        make_exp(tmp_path, config=CONFIG | {"models": ["m2"]})
    exp = open_experiment(
        tmp_path, "devsweep", CONFIG | {"max_parallel": 1}, purpose="t", protocol_ref="r",
        project_root=PROJECT_ROOT, ignore=("max_parallel",),
    )  # fmt: skip
    assert exp.name.endswith("_experiment_devsweep")


def test_sample_experiment_is_separate(tmp_path):
    full, sample = make_exp(tmp_path), make_exp(tmp_path, sample=True)
    assert full != sample and sample.name.endswith("_experiment_devsweep-sample")
    assert read_experiment(sample)["mode"] == "sample"
    with pytest.raises(ValueError, match="-sample"):
        make_exp(tmp_path, name="x-sample")


def test_manifest_matches_run_folders(tmp_path):
    exp = make_exp(tmp_path)
    rd = runs_dir(exp)
    fake_run(rd, "r_done")
    fake_run(rd, "r_partial", done=False)
    (rd / "batch").mkdir()
    (rd / "batch" / "config.json").write_text(json.dumps({"run_name": "batch", "tasks": []}))  # not a run
    man = write_manifest(exp).set_index("run_name")
    assert set(man.index) == {"r_done", "r_partial"}
    assert man.loc["r_done", "status"] == "complete" and man.loc["r_partial", "status"] == "partial"
    assert man.loc["r_done", "output_sha256"] == sha256_file(rd / "r_done" / "predictions.jsonl")
    assert man.loc["r_done", "n_pred"] == 2
    assert (exp / "README.md").exists()


def test_load_refuses_sample_experiment_unlisted_run_and_sha_mismatch(tmp_path):
    sample_exp = make_exp(tmp_path, sample=True)
    fake_run(runs_dir(sample_exp), "s1", sample=2)
    write_manifest(sample_exp)
    with pytest.raises(ValueError, match="sample experiment"):
        load_experiment_runs([sample_exp], "test")

    exp = make_exp(tmp_path)
    fake_run(runs_dir(exp), "r1")
    write_manifest(exp)
    assert [r["run_name"] for r in load_experiment_runs([exp], "test")] == ["r1"]
    assert load_experiment_runs([exp], "dev") == []

    fake_run(runs_dir(exp), "r2")  # finished after the manifest was written
    with pytest.raises(ValueError, match="missing from manifest"):
        load_experiment_runs([exp], "test")
    write_manifest(exp)
    (runs_dir(exp) / "r1" / "predictions.jsonl").write_text('{"id": "1"}\n')
    with pytest.raises(ValueError, match="sha256"):
        load_experiment_runs([exp], "test")


def test_load_refuses_sample_run_in_full_experiment(tmp_path):
    exp = make_exp(tmp_path)
    fake_run(runs_dir(exp), "s1", sample=2)
    write_manifest(exp)
    with pytest.raises(ValueError, match="sample run"):
        load_experiment_runs([exp], "test")


def test_register_points_at_runs_without_touching_them(tmp_path):
    legacy = tmp_path / "final_runs_result" / "outputs"
    run = fake_run(legacy, "20261020_cmsb_binary_m1_logprob-zs-test_s42")
    before = {p.name: sha256_file(p) for p in run.iterdir()}
    exp = register_experiment(
        tmp_path, "zeroshot-frozen", legacy, created="20261020-000000", purpose="frozen", protocol_ref="docs/06",
        config={"batch": "20261020_all_all_llms_batch_s42"},
    )  # fmt: skip
    assert runs_dir(exp) == legacy.resolve()
    assert {p.name: sha256_file(p) for p in run.iterdir()} == before
    assert [r["run_name"] for r in load_experiment_runs([exp], "test")] == [run.name]
    with pytest.raises(FileExistsError):
        register_experiment(
            tmp_path,
            "zeroshot-frozen",
            legacy,
            created="20261020-000000",
            purpose="",
            protocol_ref="",
            config={},
        )


def test_sanity_table_flags_incomplete_runs(tmp_path):
    exp = make_exp(tmp_path, sample=True)
    fake_run(runs_dir(exp), "ok", sample=2)
    fake_run(runs_dir(exp), "broken", sample=2, done=False)
    table = sanity_table(exp, ["ok", "broken", "missing"]).set_index("run_name")
    assert table.loc["ok", "ok"] and not table.loc["broken", "ok"] and not table.loc["missing", "ok"]
