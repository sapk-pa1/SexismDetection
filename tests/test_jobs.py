"""Run names: the frozen 2026-10-20 names must not change; sample and variant runs get their own names."""

from src.utils.jobs import build_jobs, method_name
from src.utils.registry import LLMSpec


def spec(name: str) -> LLMSpec:
    return LLMSpec(
        name=name, repo_id=f"org/{name}", revision="abc", params="4B", licence="x", gated=False, thinking="no",
        vram_gb=10.0,
    )  # fmt: skip


def test_method_name_defaults_unchanged():
    assert method_name("logprob", False, "test", None) == "logprob-zs-test"
    assert method_name("logprob", False, "dev", 5) == "logprob-zs-dev-n5"
    assert method_name("logprob", False, "test", None, 1) == "logprob-fs1-test"
    assert method_name("generate", True, "test", None) == "generate-zs-think-test"


def test_method_name_sample_and_variant():
    assert method_name("logprob", False, "test", None, sample=5) == "logprob-zs-test-sample-n5"
    assert method_name("logprob", False, "test", None, variant="v2") == "logprob-zs-v2-test"


def test_frozen_batch_names_reproduce():
    reg = {"gemma4-e4b": spec("gemma4-e4b")}
    jobs = build_jobs(reg, ["gemma4-e4b"], [("cmsb", "binary")], ["test"], date="20261020", seed=42)
    assert [j.run_name for j in jobs] == ["20261020_cmsb_binary_gemma4-e4b_logprob-zs-test_s42"]
    assert "--experiment-dir" not in jobs[0].argv() and "--sample" not in jobs[0].argv()


def test_variants_and_sample_jobs():
    reg = {"m": spec("m")}
    jobs = build_jobs(
        reg, ["m"], [("edos", "task_b")], ["test"], date="20261024", seed=42, experiment_dir="/x/exp",
        sample=3, variants=[{}, {"prompt_variant": "v2"}, {"option_perm": 2, "subset": 300}],
        extra_args=("--max-new-tokens", "32"),
    )  # fmt: skip
    names = [j.run_name for j in jobs]
    assert names == [
        "20261024_edos_task_b_m_logprob-zs-test-sample-n3_s42",
        "20261024_edos_task_b_m_logprob-zs-v2-test-sample-n3_s42",
        "20261024_edos_task_b_m_logprob-zs-sub300-perm2-test-sample-n3_s42",
    ]
    argv = jobs[1].argv()
    assert argv[argv.index("--experiment-dir") + 1] == "/x/exp"
    assert argv[argv.index("--sample") + 1] == "3"
    assert argv[argv.index("--prompt-variant") + 1] == "v2"
    argv = jobs[2].argv()
    assert argv[argv.index("--option-perm") + 1] == "2" and argv[argv.index("--subset") + 1] == "300"
    assert argv[argv.index("--max-new-tokens") + 1] == "32"


def test_jobs_from_ids_dir(tmp_path):
    (tmp_path / "cmsb-binary-granite4.1-8b-test-fnfp.txt").write_text("1\n2\n")
    (tmp_path / "exist-task_1_1-gemma4-e4b-dev-fnfp.txt").write_text("3\n")
    (tmp_path / "edos-task_a-gemma4-e4b-test-fnfp.txt").write_text("")  # empty lists are skipped
    (tmp_path / "cmsb-binary-granite4.1-8b-test.txt").write_text("1\n")  # not an -fnfp list
    from src.utils.jobs import jobs_from_ids_dir

    reg = {"granite4.1-8b": spec("granite4.1-8b"), "gemma4-e4b": spec("gemma4-e4b")}
    jobs = jobs_from_ids_dir(reg, tmp_path, date="20261024", seed=42, backend="text", prompt_variant="X",
                             extra_args=("--max-new-tokens", "64"))  # fmt: skip
    assert [(j.dataset, j.task, j.model, j.split) for j in jobs] == [
        ("cmsb", "binary", "granite4.1-8b", "test"), ("exist", "task_1_1", "gemma4-e4b", "dev")]  # fmt: skip
    assert (
        jobs[0].run_name
        == "20261024_cmsb_binary_granite4.1-8b_text-zs-ids-cmsb-binary-granite4.1-8b-test-fnfp-X-test_s42"
    )
    argv = jobs[1].argv()
    assert argv[argv.index("--prompt-variant") + 1] == "X" and argv[argv.index("--ids-file") + 1].endswith(
        "dev-fnfp.txt"
    )
