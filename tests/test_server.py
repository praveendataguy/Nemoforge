import time

from fastapi.testclient import TestClient

from nemoforge import server
from nemoforge.events import Registry

from .fake_llm import FakeLLM


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("NF_BUILDS_DIR", str(tmp_path))
    monkeypatch.setattr(server, "llm_factory", lambda run, cfg: FakeLLM())
    return TestClient(server.app)


def _wait(client, run_id):
    for _ in range(100):
        r = client.get(f"/api/builds/{run_id}").json()
        if r["status"] != "running":
            return r
        time.sleep(0.1)
    raise AssertionError("build did not finish")


def test_build_invoke_and_reload(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    run = client.post("/api/builds", json={"goal": "check invoice totals"}).json()
    done = _wait(client, run["id"])
    assert done["status"] == "passed"
    out = client.post(f"/api/builds/{run['id']}/invoke", json={"input": "Widget 2 x 5.00"}).json()
    assert out["output"] == "Invoice total is 10.0"

    fresh = Registry()
    fresh.load(tmp_path)
    restored = fresh.runs[run["id"]]
    assert restored.status == "passed"
    assert restored.target == "check invoice totals"
    assert restored.events[-1]["stage"] == "done"


def test_access_code_and_deploy_guard(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    monkeypatch.setenv("NF_ACCESS_CODE", "s3cret")
    assert client.post("/api/builds", json={"goal": "check invoice totals"}).status_code == 401
    run = client.post("/api/builds", json={"goal": "check invoice totals"}, headers={"X-Access-Code": "s3cret"}).json()
    _wait(client, run["id"])
    r = client.post(f"/api/builds/{run['id']}/deploy", json={"registry": "cr.example/x", "dry_run": False}, headers={"X-Access-Code": "s3cret"})
    assert r.status_code == 403
    r = client.post(f"/api/builds/{run['id']}/deploy", json={"registry": "cr.example/x"}, headers={"X-Access-Code": "s3cret"})
    cmds = r.json()["commands"]
    assert cmds[-1].startswith("nebius ai endpoint create") and "--container-port 8080" in cmds[-1]


def test_logs_and_stats_come_from_stored_builds(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    run = client.post("/api/builds", json={"goal": "check invoice totals"}).json()
    _wait(client, run["id"])
    client.post(f"/api/builds/{run['id']}/invoke", json={"input": "Widget 2 x 5.00"})

    logs = [r for r in client.get("/api/logs").json() if r["build"] == run["id"]]
    kinds = [r["kind"] for r in logs]
    assert kinds[0] == "build" and "test" in kinds and kinds[-1] == "try"
    assert logs[0]["severity"] == "info" and logs[0]["reasons"]
    assert logs[-1]["input"] == "Widget 2 x 5.00"
    assert all(r["severity"] in ("critical", "high", "medium", "low", "info") for r in logs)

    s = client.get("/api/stats").json()
    assert s["builds"] >= 1 and s["model_calls"] > 0 and s["tests_graded"] > 0
    smart = s["routing"]["smart"]
    assert smart["decisions"] > 0 and smart["by_tier"]["nano"] == smart["decisions"] and smart["avg_difficulty"] == 1
    assert s["records"] == len(client.get("/api/logs").json())


def test_agent_registry_lists_built_agents(tmp_path, monkeypatch):
    monkeypatch.setenv("NF_OWNER", "Dana")
    client = _client(tmp_path, monkeypatch)
    run = client.post("/api/builds", json={"goal": "check invoice totals"}).json()
    _wait(client, run["id"])
    assert client.get("/api/config").json()["owner"] == "Dana"
    agent = next(a for a in client.get("/api/agents").json() if a["build"] == run["id"])
    assert agent["name"] == "invoice_checker" and agent["status"] == "passed"
    assert [t["name"] for t in agent["team"]] == ["parser", "writer"] and agent["tools"] == 1


def test_invoke_accepts_a_one_hour_transcript(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    run = client.post("/api/builds", json={"goal": "check invoice totals"}).json()
    _wait(client, run["id"])
    transcript = "Alice: Widget 2 x 5.00, we agreed to ship on Friday.\n" * 1200  # ~62k chars
    assert client.post(f"/api/builds/{run['id']}/invoke", json={"input": transcript}).status_code == 200
    assert client.post(f"/api/builds/{run['id']}/invoke", json={"input": "x" * 200_001}).status_code == 422


def test_agent_memory_lessons_runs_and_recall(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    run = client.post("/api/builds", json={"goal": "check invoice totals"}).json()
    _wait(client, run["id"])
    mem = client.get(f"/api/builds/{run['id']}/memory").json()
    assert mem["lessons"][0]["source"] == "optimizer" and mem["runs"] == []

    client.post(f"/api/builds/{run['id']}/invoke", json={"input": "Widget 2 x 5.00"})
    client.post(f"/api/builds/{run['id']}/invoke", json={"input": "Gadget 1 x 3.00", "use_memory": True})
    mem = client.get(f"/api/builds/{run['id']}/memory").json()
    assert mem["total_runs"] == 2 and mem["runs"][0]["input"] == "Gadget 1 x 3.00"
    lines = (tmp_path / run["id"] / "invocations.jsonl").read_text().splitlines()
    import json as _json

    first, second = (_json.loads(x) for x in lines)
    assert "memory" not in first and second["memory"] == 1
    assert second["input"] == "Gadget 1 x 3.00"  # the log keeps what the user typed, not the memory context
    assert client.get("/api/builds/nope/memory").status_code == 404
