"""Agent Foundry web app: start builds, watch them live, try the result, deploy it.

    uvicorn foundry.server:app --port 8000
"""
from __future__ import annotations

import hmac
import json
import os
import threading
import traceback
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import deploy as deployer
from .config import settings
from .events import Run, registry
from .foundry import build
from .llm import LLM, NemotronRouter
from .runtime import run_system
from .spec import SystemSpec
from .tools import _no_net_prefix

WEB = Path(__file__).resolve().parent.parent / "web"

# Swappable so tests can run the app without network access.
llm_factory: Callable[[Run | None], LLM] = lambda run: NemotronRouter(run)

app = FastAPI(title="Agent Foundry")
registry.load(settings().builds_dir)


class BuildRequest(BaseModel):
    goal: str = Field(min_length=8, max_length=2000)
    notes: str = Field(default="", max_length=4000)


class InvokeRequest(BaseModel):
    input: str = Field(min_length=1, max_length=20000)


class DeployRequest(BaseModel):
    registry: str
    subnet_id: str = ""
    project_id: str = ""
    dry_run: bool = True


def _guard(code: str | None) -> None:
    """Builds spend Token Factory credits; a public demo can require an access code."""
    expected = os.environ.get("AF_ACCESS_CODE", "")
    if expected and not hmac.compare_digest(code or "", expected):
        raise HTTPException(401, "access code required")


def _build_dir(run_id: str) -> Path:
    d = settings().builds_dir / run_id
    if not (d / "spec.json").exists():
        raise HTTPException(404, "build not finished or not found")
    return d


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/api/config")
def config() -> dict:
    cfg = settings()
    return {
        "models": {"nano": cfg.models.nano, "super": cfg.models.super, "ultra": cfg.models.ultra},
        "nebius_key": bool(cfg.nebius_api_key),
        "tavily_key": bool(cfg.tavily_api_key),
        "pass_threshold": cfg.pass_threshold,
        "max_iterations": cfg.max_iterations,
        "sandbox_no_network": bool(_no_net_prefix()),
        "access_code": bool(os.environ.get("AF_ACCESS_CODE")),
    }


@app.post("/api/builds")
def start_build(req: BuildRequest, x_access_code: str | None = Header(default=None)) -> dict:
    _guard(x_access_code)
    run = registry.new("build", req.goal)

    def work() -> None:
        try:
            build(llm_factory(run), req.goal, run, notes=req.notes)
        except Exception as err:
            run.emit("error", f"{type(err).__name__}: {err}", traceback=traceback.format_exc()[-3000:])
            run.status = "failed"

    threading.Thread(target=work, daemon=True, name=f"build-{run.id}").start()
    return run.summary()


@app.get("/api/builds")
def list_builds() -> list[dict]:
    return registry.list()


@app.get("/api/builds/{run_id}")
def get_build(run_id: str) -> dict:
    run = registry.runs.get(run_id)
    if run is None:
        raise HTTPException(404, "unknown build")
    return {**run.summary(), "events": run.events}


@app.get("/api/builds/{run_id}/events")
async def stream(run_id: str) -> StreamingResponse:
    run = registry.runs.get(run_id)
    if run is None:
        raise HTTPException(404, "unknown build")

    async def gen():
        async for e in run.stream():
            yield f"data: {json.dumps(e, default=str)}\n\n"
        yield f"data: {json.dumps({'stage': 'end', 'status': run.status})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.post("/api/builds/{run_id}/invoke")
def invoke(run_id: str, req: InvokeRequest, x_access_code: str | None = Header(default=None)) -> dict:
    _guard(x_access_code)
    spec = SystemSpec.load(_build_dir(run_id) / "spec.json")
    out = run_system(llm_factory(None), spec, req.input, settings())
    return out


@app.post("/api/builds/{run_id}/deploy")
def deploy(run_id: str, req: DeployRequest, x_access_code: str | None = Header(default=None)) -> dict:
    _guard(x_access_code)
    if not req.dry_run and not os.environ.get("AF_ALLOW_DEPLOY"):
        raise HTTPException(403, "live deploys from the web app are off; set AF_ALLOW_DEPLOY=1 or use the CLI")
    d = _build_dir(run_id)
    try:
        cmds = deployer.deploy(d, req.registry, req.subnet_id, req.project_id, dry_run=req.dry_run)
    except Exception as err:
        raise HTTPException(500, f"deploy failed: {err}")
    return {"dry_run": req.dry_run, "commands": cmds, "image_context": str(d / "image")}
