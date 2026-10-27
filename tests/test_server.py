import time

from fastapi.testclient import TestClient

from nemoforge import server
from nemoforge.events import Registry

from .fake_llm import FakeLLM


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("NF_BUILDS_DIR", str(tmp_path))
    monkeypatch.setattr(server, "llm_factory", lambda run: FakeLLM())
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
