from pathlib import Path

import pytest

from nemoforge.config import Settings
from nemoforge.events import registry
from nemoforge.forge import build
from nemoforge.llm import parse_json
from nemoforge.runtime import run_system
from nemoforge.spec import SystemSpec
from nemoforge.tools import call_tool, run_python_tool

from .fake_llm import SPEC, FakeLLM


def test_parse_json_variants():
    assert parse_json('<think>x</think>```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"a": {"b": "}"}} trailing') == {"a": {"b": "}"}}
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_sandbox_isolation():
    assert run_python_tool("def run(x):\n    return x * 2", {"x": 4})["result"] == 8
    env = run_python_tool("import os\ndef run():\n    return dict(os.environ)", {})["result"]
    assert not any("KEY" in k for k in env)
    assert "timed out" in run_python_tool("def run():\n    while True: pass", {}, timeout=2)["error"]


def test_calculator_rejects_code():
    assert call_tool(None, "calculator", {"expression": "2**10"})["result"]["result"] == 1024
    assert "error" in call_tool(None, "calculator", {"expression": "__import__('os')"})


def test_runtime_calls_tools_in_order():
    spec = SystemSpec.model_validate({**SPEC, "tools": [{**SPEC["tools"][0], "code": "def run(lines):\n    return {'total': 10.0}"}]})
    out = run_system(FakeLLM(), spec, "Widget 2 x 5.00", Settings())
    assert [t["agent"] for t in out["trace"]] == ["parser", "writer"]
    assert out["trace"][0]["steps"][0]["result"] == {"result": {"total": 10.0}}
    assert out["output"] == "Invoice total is 10.0"


def test_build_loop_improves_and_saves(tmp_path: Path):
    cfg = Settings(builds_dir=tmp_path, max_iterations=3, eval_cases=6)
    run = registry.new("build", "check invoices")
    report = build(FakeLLM(), "check invoices", run, cfg)
    assert [r["train"] for r in report["rounds"]] == [0.0, 1.0]
    assert report["holdout_score"] == 1.0
    assert run.status == "passed"
    assert (tmp_path / run.id / "spec.json").exists()
    assert SystemSpec.load(tmp_path / run.id / "spec.json").version == 2


def test_split_keeps_every_kind_in_training():
    from nemoforge.evals import split
    from nemoforge.spec import EvalCase

    kinds = ["typical", "typical", "edge", "refuse"]
    cases = [EvalCase(id=f"c{i}", input="x", criteria="y", kind=k) for i, k in enumerate(kinds)]
    train, hold = split(cases)
    assert [c.id for c in hold] == ["c1"]
    assert {c.kind for c in train} == {"typical", "edge", "refuse"}
    kinds = ["typical"] * 4 + ["edge"] * 3 + ["refuse"] * 2
    cases = [EvalCase(id=f"c{i}", input="x", criteria="y", kind=k) for i, k in enumerate(kinds)]
    train, hold = split(cases)
    assert sorted(c.kind for c in hold) == ["edge", "refuse", "typical"]
