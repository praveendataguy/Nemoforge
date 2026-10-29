import json
from types import SimpleNamespace

import pytest

from nemoforge import insights, server
from nemoforge.config import Settings
from nemoforge.events import Run
from nemoforge.llm import NemotronRouter

from .test_server import _client, _wait


class BadRequest(Exception):
    status_code = 400


class FakeCompletions:
    def __init__(self, reject_switch=False):
        self.calls = []
        self.reject_switch = reject_switch

    def create(self, **kw):
        self.calls.append(kw)
        if self.reject_switch and "extra_body" in kw:
            raise BadRequest("unknown field chat_template_kwargs")
        msg = SimpleNamespace(content="<think>check each line, then add</think>Total is 10.0", model_extra={})
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=SimpleNamespace(prompt_tokens=40, completion_tokens=20))


def _router(completions, run=None):
    r = NemotronRouter.__new__(NemotronRouter)
    r.cfg = Settings(nebius_api_key="test")
    r.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    r.run = run
    return r


@pytest.fixture(autouse=True)
def _switch_on():
    NemotronRouter._thinking_param = True
    yield
    NemotronRouter._thinking_param = True


def test_reasoning_on_for_ultra_off_for_nano_and_captured():
    comp = FakeCompletions()
    run = Run(id="r1", kind="build", target="t")
    llm = _router(comp, run)
    assert llm.chat("ultra", "sys", "user") == "Total is 10.0"
    llm.chat("nano", "sys", "user")
    assert comp.calls[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    assert comp.calls[1]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    ultra, nano = (e["data"] for e in run.events if e["stage"] == "model")
    assert ultra["thinking"] is True and ultra["thought"] == "check each line, then add" and ultra["thought_tokens"] > 0
    assert nano["thinking"] is False


def test_switch_dropped_when_endpoint_rejects_it():
    comp = FakeCompletions(reject_switch=True)
    llm = _router(comp)
    assert llm.chat("super", "sys", "user") == "Total is 10.0"
    llm.chat("super", "sys", "user")
    assert ["extra_body" in c for c in comp.calls] == [True, False, False]


def test_recall_ranks_by_meaning_and_falls_back():
    runs = [{"input": text} for text in ("weather in Paris", "invoice for widgets", "cat pictures", "invoice for gadgets")]
    vec = {"invoice": [1.0, 0.0], "weather": [0.0, 1.0], "cat": [0.1, 0.9]}

    def embed(texts):
        return [vec[next(k for k in vec if k in t)] for t in texts]

    picked, mode = insights.recall(runs, "invoice for bolts", embed, n=2)
    assert mode == "semantic" and [r["input"] for r in picked] == ["invoice for widgets", "invoice for gadgets"]

    def broken(texts):
        raise RuntimeError("no embeddings")

    picked, mode = insights.recall(runs, "invoice", broken, n=2)
    assert mode == "recent" and [r["input"] for r in picked] == ["weather in Paris", "invoice for widgets"]
    assert insights.recall(runs[:2], "invoice", embed)[1] == "recent"


def test_invoke_recalls_related_runs_with_embeddings(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(server, "embedder", lambda cfg, texts: [[1.0, 0.0] if "Widget" in t else [0.0, 1.0] for t in texts])
    run = client.post("/api/builds", json={"goal": "check invoice totals"}).json()
    _wait(client, run["id"])
    for text in ("Widget 2 x 5.00", "Gadget 1 x 3.00", "Gadget 4 x 1.00", "Gadget 2 x 2.00"):
        client.post(f"/api/builds/{run['id']}/invoke", json={"input": text})
    out = client.post(f"/api/builds/{run['id']}/invoke", json={"input": "Widget 9 x 1.00", "use_memory": True}).json()
    assert out["memory"]["mode"] == "semantic" and out["memory"]["recalled"][-1] == "Widget 2 x 5.00"
    last = json.loads((tmp_path / run["id"] / "invocations.jsonl").read_text().splitlines()[-1])
    assert last["memory_mode"] == "semantic"
    assert client.get("/api/stats").json()["reasoning"]["semantic_recalls"] == 1


def test_dotenv_loads_keys_without_overriding_real_env(tmp_path, monkeypatch):
    from nemoforge import config

    (tmp_path / ".env").write_text('# keys\nexport NF_T_A="from file"\nNF_T_B=file\n\nNF_T_C = \'quoted\'\n', encoding="utf-8")
    monkeypatch.setenv("NF_T_B", "real")
    for k in ("NF_T_A", "NF_T_C"):
        monkeypatch.delenv(k, raising=False)
    config._load_dotenv(tmp_path / ".env")
    import os

    assert (os.environ["NF_T_A"], os.environ["NF_T_B"], os.environ["NF_T_C"]) == ("from file", "real", "quoted")
    for k in ("NF_T_A", "NF_T_C"):
        monkeypatch.delenv(k)
