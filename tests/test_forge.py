import json
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


class StuckLLM(FakeLLM):
    """The Optimizer's fixes never help; only the Healer finds the real cause."""

    def __init__(self) -> None:
        super().__init__()
        self.healed = False

    def chat(self, tier, system, user, *, max_tokens=4096):
        if system.startswith("You are the Optimizer"):
            return json.dumps({"diagnosis": "Try a longer prompt.", "changes": ["longer prompt"], "agents": SPEC["agents"], "pipeline": SPEC["pipeline"]})
        if system.startswith("You are the Healer"):
            self.healed = True
            self.rounds = 1  # the grader now passes, as if the redesign fixed the cause
            assert "failed in EVERY round: c" in user and "Widget" in user
            return json.dumps({"diagnosis": "The writer never saw the tool result.", "root_causes": ["writer has no access to the total"],
                               "overfitting": False, "changes": ["pass the total to the writer"], "agents": SPEC["agents"], "pipeline": SPEC["pipeline"]})
        return super().chat(tier, system, user, max_tokens=max_tokens)


def test_healer_repairs_a_stuck_build(tmp_path: Path):
    cfg = Settings(builds_dir=tmp_path, max_iterations=2, eval_cases=6, heal_rounds=1)
    run = registry.new("build", "check invoices")
    llm = StuckLLM()
    report = build(llm, "check invoices", run, cfg)
    assert llm.healed and run.status == "passed" and report["self_healed"]
    assert [r["train"] for r in report["rounds"]] == [0.0, 0.0, 1.0]
    assert report["version"] == 3 and "healed" in report["rounds"][1]
    heal = report["healing"][0]
    assert heal["kind"] == "stuck" and heal["fixed"] and heal["before"]["train"] == 0.0 and heal["after"]["train"] == 1.0
    assert any(e["stage"] == "healer" and e["message"].startswith("Self-heal worked") for e in run.events)
    from nemoforge import insights

    s = insights.stats([run], tmp_path)
    assert s["heals"] == 1 and s["heals_fixed"] == 1


def test_no_healing_when_turned_off(tmp_path: Path):
    cfg = Settings(builds_dir=tmp_path, max_iterations=2, eval_cases=6, heal_rounds=0)
    run = registry.new("build", "check invoices")
    llm = StuckLLM()
    report = build(llm, "check invoices", run, cfg)
    assert not llm.healed and run.status == "below_target" and report["healing"] == []


class CrashOnceLLM(FakeLLM):
    """The first Architect reply is not JSON at all, so the first attempt crashes."""

    def __init__(self) -> None:
        super().__init__()
        self.designs = 0
        self.notes = ""

    def chat(self, tier, system, user, *, max_tokens=4096):
        if system.startswith("You are the Architect"):
            self.designs += 1
            if self.designs <= 2:  # chat_json retries once, so crash twice
                return "Sorry, I cannot design that."
            self.notes = user
        if system.startswith("You are the Healer"):
            return json.dumps({"diagnosis": "The Architect replied in prose.", "fix_notes": "Keep the design small.", "retry": True})
        return super().chat(tier, system, user, max_tokens=max_tokens)


def test_healer_rebuilds_after_a_crash(tmp_path: Path):
    cfg = Settings(builds_dir=tmp_path, max_iterations=3, eval_cases=6)
    run = registry.new("build", "check invoices")
    llm = CrashOnceLLM()
    report = build(llm, "check invoices", run, cfg)
    assert run.status == "passed" and report["self_healed"]
    assert report["healing"][0]["kind"] == "crash" and "Keep the design small." in llm.notes


def test_healer_does_not_retry_a_missing_key(tmp_path: Path):
    from nemoforge.healer import heal_crash

    out = heal_crash(FakeLLM(), RuntimeError("NEBIUS_API_KEY is not set."), "", [])
    assert out["retry"] is False
