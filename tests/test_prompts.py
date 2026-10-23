"""Prompt templates: v1 stays frozen, variants keep the definitions verbatim and must match the lock."""

import hashlib
import shutil
from pathlib import Path

import pytest
import yaml

from src.model.text import parse_option, parse_yes_no
from src.prompts.builder import TEMPLATE_DIR, VARIANT_LOCK, PromptTemplate
from src.schemas import get_schema

TASKS = [("cmsb", "binary"), ("edos", "task_a"), ("edos", "task_b"), ("edos", "task_c"),
         ("exist", "task_1_1"), ("exist", "task_1_2"), ("exist", "task_1_3")]  # fmt: skip


def variants_of(prompt_id: str) -> list[str]:
    return sorted(p.name.split(".")[-2] for p in TEMPLATE_DIR.glob(f"{prompt_id}.*.yaml"))


@pytest.mark.parametrize(("dataset", "task"), TASKS)
def test_variants_keep_definitions_and_load(dataset, task):
    schema = get_schema(dataset, task)
    v1 = PromptTemplate.load(schema)
    found = variants_of(schema.prompt_id())
    assert {"T1", "T2", "T3", "T4"} <= set(found)
    for v in found:
        t = PromptTemplate.load(schema, variant=v)
        assert t.labels == v1.labels, f"{v}: definitions differ from v1"
        assert t.prompt_id == f"{schema.prompt_id()}.{v}" and t.variant == v
        msgs = (
            t.option_messages("POST")
            if schema.task_type == "multiclass"
            else t.yes_no_messages("POST", schema.labels()[0])
        )
        assert "POST" in msgs[-1]["content"]
        if v == "v2":
            assert t.logprob_prefill == "Answer: " and msgs == v1.option_messages("POST")
        else:
            assert t.logprob_prefill == ""


@pytest.mark.parametrize(("dataset", "task"), TASKS)
def test_v1_matches_the_frozen_runs(dataset, task):
    # The frozen 2026-10-20 predictions store the v1 file sha256; the loader must still produce it
    import json

    run = Path(
        f"outputs/final_runs_result/outputs/20261020_{dataset}_{task}_gemma4-e4b_logprob-zs-test_s42/predictions.jsonl"
    )
    if not run.exists():
        pytest.skip("frozen run outputs not present")
    stored = json.loads(run.open().readline())["prompt_sha256"]
    assert PromptTemplate.load(get_schema(dataset, task)).sha256 == stored


def test_lock_covers_every_variant_file():
    lock = yaml.safe_load(VARIANT_LOCK.read_text())
    files = {str(p.relative_to(TEMPLATE_DIR)) for p in TEMPLATE_DIR.glob("*/*.*.yaml")}
    assert files == set(lock)
    for name, sha in lock.items():
        assert hashlib.sha256((TEMPLATE_DIR / name).read_bytes()).hexdigest() == sha


def test_edited_variant_is_refused(tmp_path):
    tdir = tmp_path / "templates"
    shutil.copytree(TEMPLATE_DIR, tdir)
    f = tdir / "edos" / "task_b.T1.yaml"
    f.write_text(f.read_text().replace("fits this post best", "fits best"))
    with pytest.raises(ValueError, match="not frozen"):
        PromptTemplate.load(get_schema("edos", "task_b"), template_dir=tdir, variant="T1")


def test_option_rotation_keeps_letter_to_class_mapping():
    t = PromptTemplate.load(get_schema("edos", "task_b"))
    order = t.option_order(1)
    assert order == ["derogation", "animosity", "prejudiced_discussions", "threats"]
    text = t.option_messages("POST", perm=1)[-1]["content"]
    assert "A) derogation:" in text and "D) threats, plans to harm and incitement:" in text
    assert t.option_messages("POST", perm=0) == t.option_messages("POST")
    with pytest.raises(ValueError):
        t.option_order(4)


def test_text_parser_rules():
    letters = {"A": "threats", "B": "derogation", "C": "animosity", "D": "prejudiced_discussions"}
    names = {"threats": "threats, plans to harm and incitement", "derogation": "derogation",
             "animosity": "animosity", "prejudiced_discussions": "prejudiced discussions"}  # fmt: skip
    assert parse_option("B", letters, names) == ("derogation", "leading_letter")
    assert parse_option("**C)** animosity", letters, names) == ("animosity", "leading_letter")
    assert parse_option("A post that insults women.", letters, names) == (None, "unparseable")
    assert parse_option("The answer is D.", letters, names) == ("prejudiced_discussions", "keyword_letter")
    assert parse_option("This is derogation of women.", letters, names) == ("derogation", "class_name")
    assert parse_option("Both derogation and animosity.", letters, names) == (None, "unparseable")
    assert parse_yes_no("Yes. It mocks women as a group.") == (True, "It mocks women as a group.")
    assert parse_yes_no("**No** - it is a neutral question") == (False, "it is a neutral question")
    assert parse_yes_no("The post is sexist")[0] is None
