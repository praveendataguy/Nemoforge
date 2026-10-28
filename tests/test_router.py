import dataclasses

from nemoforge import router
from nemoforge.config import Settings
from nemoforge.runtime import run_system
from nemoforge.spec import AgentSpec, SystemSpec

from .fake_llm import SPEC, FakeLLM


def test_choose_follows_difficulty_but_never_drops_two_tiers():
    assert router.choose("nano", 1) == "nano"
    assert router.choose("nano", 4) == "super"
    assert router.choose("nano", 5) == "ultra"
    assert router.choose("super", 1) == "nano"
    assert router.choose("ultra", 1) == "super"  # a step planned for deep reasoning never lands on Nano
    assert router.choose("ultra", 5) == "ultra"


def _spec(tier):
    agents = [{**a, "tier": tier} for a in SPEC["agents"]]
    return SystemSpec.model_validate({**SPEC, "agents": agents, "tools": [{**SPEC["tools"][0], "code": "def run(lines):\n    return {'total': 10.0}"}]})


def test_runtime_routes_each_agent_by_request():
    llm = FakeLLM()
    short = run_system(llm, _spec("super"), "Widget 2 x 5.00", Settings())
    assert [t["tier"] for t in short["trace"]] == ["nano", "nano"]
    assert short["trace"][0]["route"] == {"agent": "parser", "planned": "super", "tier": "nano", "difficulty": 1, "why": "one short invoice line"}

    long = run_system(llm, _spec("nano"), "Widget 2 x 5.00\n" * 400, Settings())
    assert [t["tier"] for t in long["trace"]] == ["super", "super"]
    agents_tiers = [c[0] for c in llm.calls if not c[1].startswith("You are the Router")]
    assert "super" in agents_tiers and llm.calls[0][0] == "nano" and llm.calls[0][1].startswith("You are the Router")


def test_routing_off_keeps_the_architects_tier():
    cfg = dataclasses.replace(Settings(), smart_routing=False)
    out = run_system(FakeLLM(), _spec("super"), "Widget 2 x 5.00", cfg)
    assert [t["tier"] for t in out["trace"]] == ["super", "super"] and out["trace"][0]["route"] is None


class BrokenRouter(FakeLLM):
    def chat(self, tier, system, user, *, max_tokens=4096):
        if system.startswith("You are the Router"):
            return "no idea"
        return super().chat(tier, system, user, max_tokens=max_tokens)


def test_broken_rating_falls_back_to_the_plan():
    agent = AgentSpec(name="a", role="r", system_prompt="p", tier="super")
    d = router.route(BrokenRouter(), agent, "hello", [])
    assert d["tier"] == "super" and d["difficulty"] is None and "fallback" in d["why"]
