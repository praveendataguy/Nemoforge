import dataclasses
from types import SimpleNamespace

import pytest

from nemoforge.config import Settings
from nemoforge.events import registry
from nemoforge.llm import BudgetExceeded, NemotronRouter


def _router(budget, lite=False):
    cfg = dataclasses.replace(Settings(), nebius_api_key="test", budget_usd=budget, lite=lite)
    run = registry.new("build", "budget test")
    router = NemotronRouter(run, cfg)
    seen = []

    def create(model, **_):
        seen.append(model)
        usage = SimpleNamespace(prompt_tokens=100_000, completion_tokens=100_000)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))], usage=usage)

    router.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return router, run, seen


def test_cost_is_tracked_and_budget_stops_calls():
    router, run, _ = _router(budget=0.5)
    router.chat("ultra", "s", "u")  # 0.1M in * $1 + 0.1M out * $3 = $0.40
    assert run.cost_usd == pytest.approx(0.40)
    router.chat("nano", "s", "u")  # +$0.03 -> $0.43, still under
    router.chat("ultra", "s", "u")  # -> $0.83, over after this call
    with pytest.raises(BudgetExceeded):
        router.chat("nano", "s", "u")


def test_lite_mode_routes_ultra_to_super():
    router, run, seen = _router(budget=0, lite=True)
    router.chat("ultra", "s", "u")
    assert seen == [router.cfg.models.super]
    assert "ultra" not in run.usage and run.usage["super"]["calls"] == 1
