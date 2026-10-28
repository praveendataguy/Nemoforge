import json

from nemoforge import insights
from nemoforge.events import Registry


def _write(path, events):
    path.mkdir(parents=True)
    (path / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")


def test_stopped_build_is_interrupted_not_failed(tmp_path):
    start = {"seq": 0, "t": 1.0, "stage": "architect", "message": "designing", "data": {"goal": "summarise meetings"}}
    _write(tmp_path / "001-aaaaaa", [start])
    _write(tmp_path / "002-bbbbbb", [start, {"seq": 1, "t": 2.0, "stage": "error", "message": "ValueError: bad spec", "data": {}}])
    reg = Registry()
    reg.load(tmp_path)
    assert reg.runs["001-aaaaaa"].status == "interrupted"
    assert reg.runs["002-bbbbbb"].status == "failed"
    logs = {r["build"]: r for r in insights.logs(reg.runs.values(), tmp_path) if r["kind"] == "build"}
    assert logs["001-aaaaaa"]["severity"] == "medium" and "interrupted" in logs["001-aaaaaa"]["reasons"][0]["text"]
    assert logs["002-bbbbbb"]["severity"] == "critical"
    s = insights.stats(reg.runs.values(), tmp_path)
    assert s["status"]["interrupted"] == 1 and s["pass_rate"] == 0.0  # interrupted builds don't count against the pass rate


def test_routing_prices_the_same_tokens_at_ultra():
    calls = [
        {"tier": "nano", "prompt_tokens": 1_000_000, "completion_tokens": 0, "cost_usd": 0.06},
        {"tier": "super", "prompt_tokens": 0, "completion_tokens": 1_000_000, "cost_usd": 0.90},
    ]
    r = insights.routing(calls, {"ultra": (1.0, 3.0)})
    assert r["actual_usd"] == 0.96 and r["all_ultra_usd"] == 4.0
    assert r["saved_usd"] == 3.04 and round(r["saved_pct"], 2) == 0.76
    assert r["tiers"]["nano"]["calls"] == 1 and r["tiers"]["super"]["tokens"] == 1_000_000
