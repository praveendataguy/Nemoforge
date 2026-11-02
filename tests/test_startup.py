"""The server must start even when a build log or a .env value is damaged."""
import json

from nemoforge.config import Settings
from nemoforge.events import Registry


def test_load_skips_damaged_build_logs(tmp_path):
    good = tmp_path / "001-aaaaaa"
    good.mkdir()
    lines = [{"t": 1.0, "stage": "start", "data": {"goal": "summarise notes"}}, {"t": 2.0, "stage": "done", "data": {"report": {}}}]
    # A build killed mid-write leaves a half line at the end.
    (good / "events.jsonl").write_text("\n".join(json.dumps(e) for e in lines) + '\n{"t": 3.0, "sta', encoding="utf-8")

    broken = tmp_path / "002-bbbbbb"
    broken.mkdir()
    (broken / "events.jsonl").write_bytes(b"\xff\xfe not json at all\n[1, 2]\n")

    reg = Registry()
    reg.load(tmp_path)
    assert reg.runs["001-aaaaaa"].target == "summarise notes"
    assert "002-bbbbbb" not in reg.runs


def test_bad_numbers_in_env_fall_back(monkeypatch):
    monkeypatch.setenv("NF_BUDGET_USD", "$1.25")
    monkeypatch.setenv("NF_MAX_ITERATIONS", "four")
    cfg = Settings()
    assert cfg.budget_usd == 1.25
    assert cfg.max_iterations == 4
